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


"""Public OAuth refresh must keep a still-valid refresh token.

RFC 6749 section 5.1 makes ``refresh_token`` optional on a successful token
response, and section 6 says the authorization server only replaces the
previous refresh token when it issues a new one. These tests drive
``PublicClientAuth`` through its real token-endpoint client. They fail while
a refresh response that omits ``refresh_token``, or sets it to an empty
string or JSON null, drops the refresh token that was sent on that request.
"""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List
from urllib.parse import parse_qs, urlparse

import httpx

from foundry_sdk import Config, PublicClientAuth

INITIAL_REFRESH_TOKEN = "refresh-token-initial"
ROTATED_REFRESH_TOKEN = "refresh-token-rotated"
INITIAL_ACCESS_TOKEN = "access-token-initial"
REFRESHED_ACCESS_TOKEN = "access-token-refreshed"
SECOND_ACCESS_TOKEN = "access-token-second"


def _token_body(
    access_token: str,
    *,
    refresh_token: Any = None,
    include_refresh_token: bool = True,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 3600,
    }
    if include_refresh_token:
        body["refresh_token"] = refresh_token
    return body


def _initial_token_body() -> Dict[str, Any]:
    return _token_body(INITIAL_ACCESS_TOKEN, refresh_token=INITIAL_REFRESH_TOKEN)


class _TokenEndpointHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        form = parse_qs(raw_body.decode("utf-8"))
        server: "_TokenEndpoint" = self.server  # type: ignore[assignment]
        server.requests.append(
            {
                "path": self.path,
                "grant_type": form.get("grant_type", [None])[0],
                "refresh_token": form.get("refresh_token", [None])[0],
            }
        )

        if self.path.endswith("/oauth2/revoke_token"):
            payload = b"{}"
        else:
            payload = server.responses.pop(0)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _TokenEndpoint(ThreadingHTTPServer):
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        super().__init__(("127.0.0.1", 0), _TokenEndpointHandler)
        self.responses = [json.dumps(body).encode("utf-8") for body in responses]
        self.requests: List[Dict[str, Any]] = []


@contextmanager
def _running_endpoint(responses: List[Dict[str, Any]]) -> Iterator[_TokenEndpoint]:
    server = _TokenEndpoint(responses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _authenticated_client(server: _TokenEndpoint) -> PublicClientAuth:
    port = server.server_address[1]
    auth = PublicClientAuth(
        client_id="client-id-placeholder",
        redirect_url="http://127.0.0.1/callback",
        hostname=f"127.0.0.1:{port}",
        config=Config(scheme="http"),
    )
    authorize_url = auth.sign_in()
    state = parse_qs(urlparse(authorize_url).query)["state"][0]
    auth.set_token(code="authorization-code-placeholder", state=state)
    return auth


def _expire(auth: PublicClientAuth) -> None:
    assert auth._token is not None
    auth._token._expires_at = 0


def _refresh_requests(server: _TokenEndpoint) -> List[Dict[str, Any]]:
    return [request for request in server.requests if request["grant_type"] == "refresh_token"]


def _assert_refresh_kept_and_reused(server: _TokenEndpoint, auth: PublicClientAuth) -> None:
    refreshed = auth.get_token()
    assert refreshed.access_token == REFRESHED_ACCESS_TOKEN
    assert refreshed.refresh_token == INITIAL_REFRESH_TOKEN
    assert _refresh_requests(server)[-1]["path"].endswith("/multipass/api/oauth2/token")
    assert _refresh_requests(server)[-1]["refresh_token"] == INITIAL_REFRESH_TOKEN

    _expire(auth)
    minted = auth.get_token()
    assert minted.access_token == SECOND_ACCESS_TOKEN
    assert minted.refresh_token == INITIAL_REFRESH_TOKEN
    assert [request["refresh_token"] for request in _refresh_requests(server)] == [
        INITIAL_REFRESH_TOKEN,
        INITIAL_REFRESH_TOKEN,
    ]


def test_expired_get_token_keeps_omitted_refresh_token() -> None:
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, include_refresh_token=False),
        _token_body(SECOND_ACCESS_TOKEN, include_refresh_token=False),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            assert auth.get_token().refresh_token == INITIAL_REFRESH_TOKEN
            _expire(auth)
            _assert_refresh_kept_and_reused(server, auth)
        finally:
            auth.sign_out()


def test_expired_get_token_keeps_empty_refresh_token() -> None:
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, refresh_token=""),
        _token_body(SECOND_ACCESS_TOKEN, include_refresh_token=False),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            _expire(auth)
            _assert_refresh_kept_and_reused(server, auth)
        finally:
            auth.sign_out()


def test_expired_get_token_keeps_null_refresh_token() -> None:
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, refresh_token=None),
        _token_body(SECOND_ACCESS_TOKEN, include_refresh_token=False),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            _expire(auth)
            _assert_refresh_kept_and_reused(server, auth)
        finally:
            auth.sign_out()


def test_background_refresh_keeps_omitted_refresh_token() -> None:
    """The auto-refresh thread calls ``_try_refresh_token`` directly."""
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, include_refresh_token=False),
        _token_body(SECOND_ACCESS_TOKEN, include_refresh_token=False),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            assert auth._try_refresh_token() is True
            assert auth.get_token().access_token == REFRESHED_ACCESS_TOKEN
            assert auth.get_token().refresh_token == INITIAL_REFRESH_TOKEN
            _expire(auth)
            minted = auth.get_token()
            assert minted.access_token == SECOND_ACCESS_TOKEN
            assert minted.refresh_token == INITIAL_REFRESH_TOKEN
            assert [request["refresh_token"] for request in _refresh_requests(server)] == [
                INITIAL_REFRESH_TOKEN,
                INITIAL_REFRESH_TOKEN,
            ]
        finally:
            auth.sign_out()


def test_unauthorized_retry_keeps_omitted_refresh_token() -> None:
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, include_refresh_token=False),
        _token_body(SECOND_ACCESS_TOKEN, include_refresh_token=False),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            calls = {"count": 0}

            def call_foundry(token: Any) -> str:
                calls["count"] += 1
                if calls["count"] == 1:
                    request = httpx.Request("GET", "https://example.invalid/foundry")
                    response = httpx.Response(401, request=request)
                    raise httpx.HTTPStatusError(
                        "unauthorized", request=request, response=response
                    )
                assert token.access_token == REFRESHED_ACCESS_TOKEN
                return "ok"

            assert auth.execute_with_token(call_foundry) == "ok"
            assert calls["count"] == 2
            assert auth.get_token().refresh_token == INITIAL_REFRESH_TOKEN
            assert _refresh_requests(server)[-1]["refresh_token"] == INITIAL_REFRESH_TOKEN

            _expire(auth)
            minted = auth.get_token()
            assert minted.access_token == SECOND_ACCESS_TOKEN
            assert minted.refresh_token == INITIAL_REFRESH_TOKEN
            assert [request["refresh_token"] for request in _refresh_requests(server)] == [
                INITIAL_REFRESH_TOKEN,
                INITIAL_REFRESH_TOKEN,
            ]
        finally:
            auth.sign_out()


def test_non_empty_refresh_token_replaces_the_previous_one() -> None:
    responses = [
        _initial_token_body(),
        _token_body(REFRESHED_ACCESS_TOKEN, refresh_token=ROTATED_REFRESH_TOKEN),
        _token_body(SECOND_ACCESS_TOKEN, refresh_token=ROTATED_REFRESH_TOKEN),
    ]
    with _running_endpoint(responses) as server:
        auth = _authenticated_client(server)
        try:
            _expire(auth)
            refreshed = auth.get_token()
            assert refreshed.access_token == REFRESHED_ACCESS_TOKEN
            assert refreshed.refresh_token == ROTATED_REFRESH_TOKEN

            _expire(auth)
            minted = auth.get_token()
            assert minted.access_token == SECOND_ACCESS_TOKEN
            assert minted.refresh_token == ROTATED_REFRESH_TOKEN
            assert [request["refresh_token"] for request in _refresh_requests(server)] == [
                INITIAL_REFRESH_TOKEN,
                ROTATED_REFRESH_TOKEN,
            ]
        finally:
            auth.sign_out()
