#  Copyright 2024 Palantir Technologies, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""Public OAuth sign-out must stick while a refresh is in flight.

The refresh request is already inside HttpClient.post when sign_out runs.
These tests deliver that response only after sign_out has returned, or, when
sign_out waits for the refresh to finish, after sign_out has blocked. A later
API call must not send the access token from that response.

ConfidentialClientAuth is not covered. Its get_token fetches a new token with
the stored client secret whenever the in-memory token is missing, so sign-out
does not stick even without this race.
"""

import threading
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from foundry_sdk import PublicClientAuth
from foundry_sdk._core.api_client import ApiClient, RequestInfo
from foundry_sdk._core.http_client import HttpClient
from foundry_sdk._errors.not_authenticated import NotAuthenticated

HOST = "https://foundry.example.test"
CLIENT_ID = "client-id"
REDIRECT_URL = "https://app.example.test/callback"
# Obvious placeholders. These are not credentials.
AUTHORIZATION_CODE = "authorization-code-placeholder"
ORIGINAL_ACCESS_TOKEN = "access-token-before-sign-out"
REFRESH_TOKEN = "refresh-token-placeholder"
REFRESHED_ACCESS_TOKEN = "access-token-from-inflight-refresh"
REFRESHED_REFRESH_TOKEN = "refresh-token-from-inflight-refresh"


class _Gate:
    def __init__(self) -> None:
        self.refresh_entered = threading.Event()
        self.release_refresh = threading.Event()
        self.refresh_succeeded = threading.Event()
        self.refresh_finished = threading.Event()
        self.refresh_bodies: List[Dict[str, List[str]]] = []
        self.revoke_bodies: List[Dict[str, List[str]]] = []
        self.api_authorizations: List[Optional[str]] = []
        self.refresh_on_daemon: Optional[bool] = None


def _form(request: httpx.Request) -> Dict[str, List[str]]:
    return parse_qs(request.content.decode("utf-8"), keep_blank_values=True)


def _token_response(access_token: str, expires_in: int, refresh_token: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "refresh_token": refresh_token,
        },
    )


def _install_transport(gate: _Gate, initial_expires_in: int) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/oauth2/token"):
            form = _form(request)
            grant_type = form.get("grant_type", [""])[0]
            if grant_type == "authorization_code":
                return _token_response(ORIGINAL_ACCESS_TOKEN, initial_expires_in, REFRESH_TOKEN)
            if grant_type == "refresh_token":
                gate.refresh_bodies.append(form)
                gate.refresh_on_daemon = threading.current_thread().daemon
                gate.refresh_entered.set()
                if not gate.release_refresh.wait(10):
                    return httpx.Response(500, json={"error": "refresh was not released"})
                gate.refresh_succeeded.set()
                return _token_response(REFRESHED_ACCESS_TOKEN, 3600, REFRESHED_REFRESH_TOKEN)
            return httpx.Response(400, json={"error": "unexpected grant"})
        if path.endswith("/oauth2/revoke_token"):
            gate.revoke_bodies.append(_form(request))
            return httpx.Response(200, json={})
        if path.startswith("/api/"):
            gate.api_authorizations.append(request.headers.get("authorization"))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "unexpected path", "path": path})

    return httpx.MockTransport(handler)


def _signed_in_auth(should_refresh: bool, initial_expires_in: int) -> tuple:
    gate = _Gate()
    transport = _install_transport(gate, initial_expires_in)
    auth = PublicClientAuth(
        client_id=CLIENT_ID,
        redirect_url=REDIRECT_URL,
        hostname=HOST,
        should_refresh=should_refresh,
    )
    auth._client = HttpClient()
    auth._client._transport = transport

    original_try_refresh = auth._try_refresh_token

    def try_refresh_and_signal() -> bool:
        try:
            return original_try_refresh()
        finally:
            gate.refresh_finished.set()

    auth._try_refresh_token = try_refresh_and_signal

    state = parse_qs(urlparse(auth.sign_in()).query)["state"][0]
    auth.set_token(AUTHORIZATION_CODE, state)
    return auth, gate, transport


def _sign_out_during_refresh(auth: PublicClientAuth, gate: _Gate) -> None:
    """Let sign_out run while the refresh post is blocked inside the transport."""
    assert gate.refresh_entered.wait(5), "refresh did not reach the token endpoint"
    assert gate.refresh_bodies, "refresh request body was not captured"
    refresh_body = gate.refresh_bodies[0]
    assert refresh_body["grant_type"] == ["refresh_token"]
    assert refresh_body["refresh_token"] == [REFRESH_TOKEN]
    assert "access_token" not in refresh_body

    signer = threading.Thread(target=auth.sign_out, name="sign-out")
    signer.start()
    signer.join(timeout=1)
    if signer.is_alive():
        # sign_out is waiting for the in-flight refresh to finish.
        gate.release_refresh.set()
        signer.join(timeout=5)
        assert not signer.is_alive(), "sign_out did not finish after the refresh returned"
    else:
        # sign_out returned while the refresh response was still outstanding.
        gate.release_refresh.set()


def _assert_signed_out(auth: PublicClientAuth, gate: _Gate, transport: httpx.BaseTransport) -> None:
    assert gate.refresh_finished.wait(5), "in-flight refresh did not return to the caller"
    assert gate.refresh_succeeded.is_set(), "in-flight refresh did not return a token response"

    api = ApiClient(auth=auth, hostname=HOST)
    api._session._transport = transport
    request = RequestInfo.with_defaults(
        "GET",
        "/example",
        response_type=None,
    )
    try:
        api.call_api(request)
    except NotAuthenticated:
        pass
    else:
        pytest.fail(
            "API call after sign_out was sent with Authorization "
            f"{gate.api_authorizations!r}; revoke requests were {gate.revoke_bodies!r}"
        )
    assert gate.api_authorizations == []


def test_background_refresh_after_sign_out_is_not_used() -> None:
    # expires_in 0 makes the daemon enter refresh without waiting out a lifetime.
    auth, gate, transport = _signed_in_auth(should_refresh=True, initial_expires_in=0)
    _sign_out_during_refresh(auth, gate)
    assert gate.refresh_on_daemon is True
    _assert_signed_out(auth, gate, transport)


def test_get_token_refresh_after_sign_out_is_not_used() -> None:
    # 60s expiry padding makes a 1s token already expired, so get_token refreshes.
    auth, gate, transport = _signed_in_auth(should_refresh=False, initial_expires_in=1)
    observed: Dict[str, Any] = {}

    def refresh_via_get_token() -> None:
        try:
            observed["access_token"] = auth.get_token().access_token
        except NotAuthenticated as exc:
            observed["error"] = exc

    worker = threading.Thread(target=refresh_via_get_token, name="get-token")
    worker.start()
    _sign_out_during_refresh(auth, gate)
    worker.join(timeout=5)
    assert not worker.is_alive()

    _assert_signed_out(auth, gate, transport)
    assert observed.get("access_token") != REFRESHED_ACCESS_TOKEN


def test_unauthorized_retry_refresh_after_sign_out_is_not_used() -> None:
    # Long lifetime keeps the daemon asleep and the token unexpired, so the
    # refresh under test is the 401 retry inside execute_with_token.
    auth, gate, transport = _signed_in_auth(should_refresh=True, initial_expires_in=3600)
    observed: Dict[str, Any] = {"presented": []}
    unauthorized = httpx.Response(401, request=httpx.Request("GET", HOST + "/api/example"))

    def call_with_unauthorized_retry() -> None:
        def use_token(token: Any) -> str:
            observed["presented"].append(token.access_token)
            if len(observed["presented"]) == 1:
                raise httpx.HTTPStatusError(
                    "unauthorized",
                    request=unauthorized.request,
                    response=unauthorized,
                )
            return "called"

        try:
            observed["result"] = auth.execute_with_token(use_token)
        except Exception as exc:
            observed["error"] = exc

    worker = threading.Thread(target=call_with_unauthorized_retry, name="api-call")
    worker.start()
    _sign_out_during_refresh(auth, gate)
    worker.join(timeout=5)
    assert not worker.is_alive()

    _assert_signed_out(auth, gate, transport)
    assert REFRESHED_ACCESS_TOKEN not in observed["presented"]
