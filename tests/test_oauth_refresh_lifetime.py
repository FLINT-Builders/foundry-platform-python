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


"""OAuth refresh must not spin when the token endpoint reports a non-positive lifetime.

These tests exercise ConfidentialClientAuth and PublicClientAuth against a local token
endpoint. They fail while a non-positive expires_in is stored and the auto-refresh thread
waits for three quarters of that lifetime, and they pass once that response is rejected
and the refresh thread does not keep calling the token endpoint.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import pydantic
import pytest

from foundry_sdk import ConfidentialClientAuth, Config, PublicClientAuth
from foundry_sdk._errors.not_authenticated import NotAuthenticated

_ACCEPTED_REJECTION = (NotAuthenticated, pydantic.ValidationError, ValueError)
_CLIENT_ID = "client-id-placeholder"
_CLIENT_SECRET = "client-secret-placeholder"
_REFRESH_TOKEN = "refresh-token-placeholder"
_ACCESS_TOKEN = "access-token-placeholder"
_AUTH_CODE = "authorization-code-placeholder"
_REDIRECT_URL = "http://127.0.0.1/callback"


class _Script:
    def __init__(self, lifetimes: Sequence[int], include_refresh_token: bool) -> None:
        self._lifetimes = list(lifetimes)
        self.include_refresh_token = include_refresh_token
        self._lock = threading.Lock()
        self.calls: List[dict] = []

    def next_lifetime(self) -> int:
        with self._lock:
            if len(self._lifetimes) > 1:
                return self._lifetimes.pop(0)
            return self._lifetimes[0]

    def add(self, call: dict) -> None:
        with self._lock:
            self.calls.append(call)

    def token_calls(self) -> List[dict]:
        with self._lock:
            return [call for call in self.calls if call["kind"] == "token"]


def _start_server(script: _Script) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode())
            kind = "revoke" if self.path.endswith("/oauth2/revoke_token") else "token"
            script.add(
                {
                    "kind": kind,
                    "grant_type": (form.get("grant_type") or [None])[0],
                    "has_client_secret": "client_secret" in form,
                    "has_refresh_token": "refresh_token" in form,
                }
            )
            if kind == "revoke":
                body = b""
            else:
                payload = {
                    "access_token": _ACCESS_TOKEN,
                    "token_type": "bearer",
                    "expires_in": script.next_lifetime(),
                }
                if script.include_refresh_token:
                    payload["refresh_token"] = _REFRESH_TOKEN
                body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stop(auth: Optional[object], server: ThreadingHTTPServer) -> None:
    if auth is not None:
        try:
            auth.sign_out()  # type: ignore[attr-defined]
        except Exception:
            pass
    server.shutdown()
    server.server_close()


def _config() -> Config:
    return Config(scheme="http", timeout=2)


def _confidential(server: ThreadingHTTPServer) -> ConfidentialClientAuth:
    host, port = server.server_address
    return ConfidentialClientAuth(
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
        hostname=f"{host}:{port}",
        config=_config(),
    )


def _public(server: ThreadingHTTPServer) -> PublicClientAuth:
    host, port = server.server_address
    return PublicClientAuth(
        client_id=_CLIENT_ID,
        redirect_url=_REDIRECT_URL,
        hostname=f"{host}:{port}",
        config=_config(),
    )


def _sign_in_state(auth: PublicClientAuth) -> str:
    return parse_qs(urlparse(auth.sign_in()).query)["state"][0]


def _describe(calls: List[dict]) -> str:
    grants: dict = {}
    for call in calls:
        grants[call["grant_type"]] = grants.get(call["grant_type"], 0) + 1
    secret_fields = sum(1 for call in calls if call["has_client_secret"])
    refresh_fields = sum(1 for call in calls if call["has_refresh_token"])
    return (
        f"token_requests={len(calls)} grants={grants} "
        f"requests_with_client_secret_field={secret_fields} "
        f"requests_with_refresh_token_field={refresh_fields}"
    )


@pytest.mark.parametrize("lifetime", [0, -1])
def test_confidential_client_rejects_non_positive_expires_in(lifetime: int) -> None:
    script = _Script([lifetime], include_refresh_token=False)
    server = _start_server(script)
    auth: Optional[ConfidentialClientAuth] = None
    try:
        auth = _confidential(server)
        rejected: Optional[BaseException] = None
        try:
            auth.get_token()
        except _ACCEPTED_REJECTION as exc:
            rejected = exc
        time.sleep(0.5)
        calls = script.token_calls()
        assert rejected is not None and len(calls) == 1, (
            "non-positive expires_in must be rejected before automatic refresh repeats "
            f"the token request ({_describe(calls)}, rejected={rejected is not None})"
        )
        assert calls[0]["grant_type"] == "client_credentials"
        assert calls[0]["has_client_secret"]
    finally:
        _stop(auth, server)


@pytest.mark.parametrize("lifetime", [0, -1])
def test_public_client_rejects_non_positive_expires_in(lifetime: int) -> None:
    script = _Script([lifetime], include_refresh_token=True)
    server = _start_server(script)
    auth: Optional[PublicClientAuth] = None
    try:
        auth = _public(server)
        state = _sign_in_state(auth)
        rejected: Optional[BaseException] = None
        try:
            auth.set_token(code=_AUTH_CODE, state=state)
        except _ACCEPTED_REJECTION as exc:
            rejected = exc
        time.sleep(0.5)
        calls = script.token_calls()
        assert rejected is not None and len(calls) == 1, (
            "non-positive expires_in must be rejected before automatic refresh repeats "
            f"the token request ({_describe(calls)}, rejected={rejected is not None})"
        )
        assert calls[0]["grant_type"] == "authorization_code"
    finally:
        _stop(auth, server)


def test_public_client_without_refresh_token_does_not_spin() -> None:
    script = _Script([0], include_refresh_token=False)
    server = _start_server(script)
    auth: Optional[PublicClientAuth] = None
    try:
        auth = _public(server)
        state = _sign_in_state(auth)
        rejected: Optional[BaseException] = None
        cpu_before = time.process_time()
        try:
            auth.set_token(code=_AUTH_CODE, state=state)
        except _ACCEPTED_REJECTION as exc:
            rejected = exc
        time.sleep(0.5)
        cpu_used = time.process_time() - cpu_before
        calls = script.token_calls()
        assert rejected is not None and len(calls) == 1 and cpu_used < 0.2, (
            "expires_in of 0 must not leave the refresh thread running "
            f"(cpu_seconds={cpu_used:.3f}, {_describe(calls)}, rejected={rejected is not None})"
        )
    finally:
        _stop(auth, server)


@pytest.mark.parametrize("client_kind", ["confidential", "public"])
def test_refresh_response_with_zero_expires_in_does_not_loop(client_kind: str) -> None:
    """A later token response can replace a positive lifetime with zero.

    The first response uses a one-second lifetime so the background wait is short enough
    to observe. After that response, every token response uses expires_in 0.
    """
    script = _Script([1, 0], include_refresh_token=True)
    server = _start_server(script)
    auth: Optional[object] = None
    try:
        if client_kind == "confidential":
            auth = _confidential(server)
            auth.get_token()
        else:
            auth = _public(server)
            auth.set_token(code=_AUTH_CODE, state=_sign_in_state(auth))
        time.sleep(1.5)
        count_after_refresh_window = len(script.token_calls())
        time.sleep(0.5)
        calls = script.token_calls()
        growth = len(calls) - count_after_refresh_window
        assert len(calls) <= 3 and growth <= 1, (
            "a refresh response with expires_in 0 must not keep requesting tokens "
            f"(growth_in_0.5s={growth}, {_describe(calls)})"
        )
    finally:
        _stop(auth, server)


def test_public_get_token_rejects_zero_expires_in_from_refresh() -> None:
    """PublicClientAuth.get_token refreshes an expired token and then restarts auto-refresh."""
    script = _Script([61, 0], include_refresh_token=True)
    server = _start_server(script)
    auth: Optional[PublicClientAuth] = None
    try:
        auth = _public(server)
        auth.set_token(code=_AUTH_CODE, state=_sign_in_state(auth))
        time.sleep(1.5)
        rejected: Optional[BaseException] = None
        try:
            auth.get_token()
        except _ACCEPTED_REJECTION as exc:
            rejected = exc
        time.sleep(0.5)
        calls = script.token_calls()
        refresh_calls = [call for call in calls if call["grant_type"] == "refresh_token"]
        assert rejected is not None and len(refresh_calls) == 1 and len(calls) == 2, (
            "get_token must surface an error when the refreshed lifetime is zero and must not "
            f"keep refreshing ({_describe(calls)}, rejected={rejected is not None})"
        )
        assert refresh_calls[0]["has_refresh_token"]
    finally:
        _stop(auth, server)


def test_confidential_get_token_rejects_zero_expires_in_from_refresh() -> None:
    """ConfidentialClientAuth.get_token fetches a new token when the stored one is expired."""
    script = _Script([61, 0], include_refresh_token=False)
    server = _start_server(script)
    auth: Optional[ConfidentialClientAuth] = None
    try:
        auth = _confidential(server)
        auth.get_token()
        time.sleep(1.5)
        rejected: Optional[BaseException] = None
        try:
            auth.get_token()
        except _ACCEPTED_REJECTION as exc:
            rejected = exc
        time.sleep(0.5)
        calls = script.token_calls()
        assert rejected is not None and len(calls) == 2, (
            "get_token must surface an error when the refreshed lifetime is zero and must not "
            f"keep requesting tokens ({_describe(calls)}, rejected={rejected is not None})"
        )
        assert all(call["grant_type"] == "client_credentials" for call in calls)
        assert all(call["has_client_secret"] for call in calls)
    finally:
        _stop(auth, server)


def test_positive_expires_in_issues_one_token_request() -> None:
    script = _Script([3600], include_refresh_token=True)
    server = _start_server(script)
    confidential: Optional[ConfidentialClientAuth] = None
    try:
        confidential = _confidential(server)
        token = confidential.get_token()
        assert token.expires_in == 3600
        time.sleep(0.4)
        calls = script.token_calls()
        assert len(calls) == 1, _describe(calls)
        assert calls[0]["grant_type"] == "client_credentials"
        assert calls[0]["has_client_secret"]
    finally:
        _stop(confidential, server)
