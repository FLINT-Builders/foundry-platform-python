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


"""Default HTTP timeouts must stay finite and must be enforced on the wire.

These tests fail while ``Config.timeout`` is unset. httpx treats an explicit
``None`` timeout as no connect, read, or write limit, and every client built
from the default config then waits until the peer finishes. A peer that
accepts the TCP connection and withholds the rest of the handshake, or sends
response headers and withholds the body, holds the calling thread.

Generated resource methods all go through ``ApiClient.call_api`` or
``AsyncApiClient.call_api``. The calls below are one method from each
Foundry client generation, plus the OAuth and language-model clients that
build their own ``HttpClient``. Other generated methods, SSE iteration, and
paged iterators are not called individually; they share those two call paths.
A write that fills the socket buffer is not given its own stalled peer. A
small request fits in the buffer and then blocks in the read, which the body
stall covers. Connect, read, and write on the default timeout object still
all have to be finite.
"""

import asyncio
import os
import socket
import threading
import time
from typing import Callable, Optional, Sequence, Tuple

import httpx

from foundry_sdk import (
    ConfidentialClientAuth,
    Config,
    PublicClientAuth,
    UserTokenAuth,
)
from foundry_sdk._core.context_and_environment_vars import TOKEN_VAR
from foundry_sdk._core.http_client import AsyncHttpClient, HttpClient
from foundry_sdk._core.oauth_utils import OAuthToken, OAuthTokenResponse
from foundry_sdk._errors import ConnectTimeout as SdkConnectTimeout
from foundry_sdk._errors import ReadTimeout as SdkReadTimeout
from foundry_sdk._errors import WriteTimeout as SdkWriteTimeout
from foundry_sdk.v1.client import AsyncFoundryClient as AsyncFoundryClientV1
from foundry_sdk.v1.client import FoundryClient as FoundryClientV1
from foundry_sdk.v2.client import AsyncFoundryClient as AsyncFoundryClientV2
from foundry_sdk.v2.client import FoundryClient as FoundryClientV2
from foundry_sdk.v2.language_models.utils import get_async_http_client, get_http_client
from foundry_sdk.v3.client import AsyncFoundryClient as AsyncFoundryClientV3
from foundry_sdk.v3.client import FoundryClient as FoundryClientV3

# Placeholder credential material. These values are not real secrets.
_TOKEN = "placeholder-token"
_CLIENT_ID = "placeholder-client-id"
_CLIENT_SECRET = "placeholder-client-secret"
_REFRESH_TOKEN = "placeholder-refresh-token"
_AUTH_CODE = "placeholder-authorization-code"
_ONTOLOGY_RID = "ri.foundry.main.ontology.placeholder"
_ENROLLMENT_RID = "ri.foundry.main.enrollment.placeholder"
_ENDPOINT_SET_RID = "ri.foundry.main.endpoint-set.placeholder"

# A default at this scale matches a bounded connect/read timeout. Longer than
# this still lets one withheld response occupy a worker for too long.
_MAX_DEFAULT_SECONDS = 60.0
# Used only when the client timeout component is missing, so the test can
# observe that the call is still blocked. A finite default waits for that
# default plus this grace period.
_UNBOUNDED_WATCHDOG_SECONDS = 3.0
_GRACE_SECONDS = 5.0

_CONNECT_TIMEOUTS = (SdkConnectTimeout, httpx.ConnectTimeout)
_READ_TIMEOUTS = (SdkReadTimeout, httpx.ReadTimeout)
_WRITE_TIMEOUTS = (SdkWriteTimeout, httpx.WriteTimeout)

_PROXY_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def _timeout_limits(
    timeout: object,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return connect, read, and write limits. ``None`` means that limit is off."""
    if isinstance(timeout, httpx.Timeout):
        return timeout.connect, timeout.read, timeout.write
    if timeout is None:
        return None, None, None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise AssertionError(f"unexpected timeout value {timeout!r}")
    value = float(timeout)
    return value, value, value


def _assert_finite_limits(timeout: object, where: str) -> Tuple[float, float, float]:
    connect, read, write = _timeout_limits(timeout)
    for name, value in (("connect", connect), ("read", read), ("write", write)):
        assert value is not None, f"{where} {name} timeout is disabled"
        assert value > 0, f"{where} {name} timeout must be positive, got {value}"
        assert (
            value <= _MAX_DEFAULT_SECONDS
        ), f"{where} {name} timeout {value}s exceeds {_MAX_DEFAULT_SECONDS}s"
    assert connect is not None and read is not None and write is not None
    return connect, read, write


class _StallingPeer:
    """Accept TCP connections and never finish an HTTP response.

    A TLS client hello is left unanswered, so the handshake never completes.
    An HTTP request is answered with headers and then no body.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._connections: list[socket.socket] = []
        self._lock = threading.Lock()
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind(("127.0.0.1", 0))
        self._listen.listen(128)
        self.port = self._listen.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def accepted(self) -> int:
        with self._lock:
            return len(self._connections)

    def _accept_loop(self) -> None:
        self._listen.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _addr = self._listen.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self._connections.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(0.5)
            first = b""
            while not first and not self._stop.is_set():
                try:
                    first = conn.recv(1)
                except socket.timeout:
                    continue
            if not first or self._stop.is_set():
                return
            # TLS records start with content type 0x16 (handshake). Withhold it.
            if first == b"\x16":
                self._stop.wait()
                return
            data = bytearray(first)
            while b"\r\n\r\n" not in data and not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    return
                data.extend(chunk)
            if self._stop.is_set():
                return
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 32\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            self._stop.wait()
        except OSError:
            return

    def close(self) -> None:
        self._stop.set()
        try:
            self._listen.close()
        except OSError:
            pass
        with self._lock:
            connections = list(self._connections)
        for conn in connections:
            try:
                conn.close()
            except OSError:
                pass
        self._thread.join(timeout=2)


def _clear_proxy_env() -> None:
    for name in _PROXY_ENV:
        os.environ.pop(name, None)


def _auth() -> UserTokenAuth:
    return UserTokenAuth(token=_TOKEN)


def _phase_limit(phase: str) -> float:
    client = HttpClient()
    try:
        connect, read, write = _timeout_limits(client.timeout)
    finally:
        client.close()
    value = {"connect": connect, "read": read, "write": write}[phase]
    if value is None:
        return _UNBOUNDED_WATCHDOG_SECONDS
    return float(value) + _GRACE_SECONDS


def test_default_timeout_is_finite_on_config_and_both_clients() -> None:
    """The default config and both HTTP clients must share a bounded timeout."""
    expected = _assert_finite_limits(Config().timeout, "Config")

    sync_client = HttpClient()
    try:
        sync_limits = _assert_finite_limits(sync_client.timeout, "HttpClient")
    finally:
        sync_client.close()

    async_client = AsyncHttpClient()
    try:
        async_limits = _assert_finite_limits(async_client.timeout, "AsyncHttpClient")
    finally:
        asyncio.run(async_client.aclose())

    assert sync_limits == expected, (sync_limits, expected)
    assert async_limits == expected, (async_limits, expected)


def test_explicit_timeout_none_still_disables_limits() -> None:
    """Callers can still opt out by passing ``timeout=None`` on the config."""
    config = Config(timeout=None)
    sync_client = HttpClient(config)
    try:
        assert _timeout_limits(sync_client.timeout) == (None, None, None)
    finally:
        sync_client.close()

    async_client = AsyncHttpClient(config)
    try:
        assert _timeout_limits(async_client.timeout) == (None, None, None)
    finally:
        asyncio.run(async_client.aclose())


def test_stalling_peer_cannot_block_default_client_calls() -> None:
    """A peer that accepts and withholds the response must not hold the caller."""
    assert socket.getdefaulttimeout() is None
    _clear_proxy_env()
    TOKEN_VAR.set(_TOKEN)

    peer = _StallingPeer()
    host = f"127.0.0.1:{peer.port}"
    http_config = Config(scheme="http")
    failures: list[str] = []
    workers: list[threading.Thread] = []

    try:
        calls = _calls(host, http_config)
        limits = {name: _phase_limit(phase) for name, _call, phase in calls}
        expected = {
            "connect": _CONNECT_TIMEOUTS,
            "read": _READ_TIMEOUTS,
            "write": _WRITE_TIMEOUTS,
        }

        boxes: dict[str, dict[str, BaseException]] = {}
        phases: dict[str, str] = {}
        for name, call, phase in calls:
            box: dict[str, BaseException] = {}
            boxes[name] = box
            phases[name] = phase

            def target(
                call: Callable[[], None] = call, box: dict[str, BaseException] = box
            ) -> None:
                try:
                    call()
                except BaseException as exc:  # noqa: BLE001 - the assertion checks the type
                    box["exc"] = exc

            worker = threading.Thread(target=target, name=name, daemon=True)
            workers.append(worker)

        started = time.monotonic()
        for worker in workers:
            worker.start()

        deadline = started + max(limits.values())
        while time.monotonic() < deadline:
            if all(not worker.is_alive() for worker in workers):
                break
            time.sleep(0.05)

        for worker in workers:
            limit = limits[worker.name]
            phase = phases[worker.name]
            if worker.is_alive():
                failures.append(
                    f"{worker.name}: still blocked after {limit:.1f}s "
                    f"while waiting on the {phase} limit"
                )
                continue
            exc = boxes[worker.name].get("exc")
            if exc is None:
                failures.append(f"{worker.name}: returned a response from the stalling peer")
                continue
            if not isinstance(exc, expected[phase]):
                failures.append(
                    f"{worker.name}: {type(exc).__module__}.{type(exc).__name__}: {exc}"
                )
    finally:
        peer.close()
        for worker in workers:
            worker.join(timeout=2)

    assert peer.accepted > 0, "the stalling peer never accepted a connection"
    assert not failures, "stalling peer blocked or failed calls:\n" + "\n".join(failures)


def _calls(host: str, http_config: Config) -> Sequence[Tuple[str, Callable[[], None], str]]:
    auth = _auth()
    v1 = FoundryClientV1(auth=auth, hostname=host)
    v2 = FoundryClientV2(auth=auth, hostname=host)
    v3 = FoundryClientV3(auth=auth, hostname=host)
    v1_http = FoundryClientV1(auth=auth, hostname=host, config=http_config)
    v2_http = FoundryClientV2(auth=auth, hostname=host, config=http_config)
    v2_stream_client = FoundryClientV2(auth=auth, hostname=host, config=http_config)
    async_v1 = AsyncFoundryClientV1(auth=auth, hostname=host, preview=True)
    async_v2 = AsyncFoundryClientV2(auth=auth, hostname=host, preview=True)
    async_v3 = AsyncFoundryClientV3(auth=auth, hostname=host, preview=True)
    async_v2_http = AsyncFoundryClientV2(auth=auth, hostname=host, config=http_config, preview=True)
    confidential = ConfidentialClientAuth(
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
        hostname=host,
        should_refresh=False,
    )
    public = PublicClientAuth(
        client_id=_CLIENT_ID,
        redirect_url="http://127.0.0.1/callback",
        hostname=host,
        should_refresh=False,
    )
    public.sign_in()
    assert public._auth_request is not None
    public_state = public._auth_request.state
    public_refresh = PublicClientAuth(
        client_id=_CLIENT_ID,
        redirect_url="http://127.0.0.1/callback",
        hostname=host,
        should_refresh=False,
    )
    public_refresh._token = OAuthToken(
        token=OAuthTokenResponse(
            token_response={
                "access_token": _TOKEN,
                "token_type": "bearer",
                "expires_in": 0,
                "refresh_token": _REFRESH_TOKEN,
            }
        )
    )
    confidential_revoke = ConfidentialClientAuth(
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
        hostname=host,
        should_refresh=False,
    )
    confidential_revoke._token = OAuthToken(
        token=OAuthTokenResponse(
            token_response={
                "access_token": _TOKEN,
                "token_type": "bearer",
                "expires_in": 3600,
            }
        )
    )
    sync_lm = get_http_client(preview=True)
    async_lm = get_async_http_client(preview=True)

    def v1_get() -> None:
        v1.ontologies.Ontology.get(_ONTOLOGY_RID)

    def v2_get() -> None:
        v2.admin.Enrollment.get(_ENROLLMENT_RID)

    def v3_get() -> None:
        v3.endpoints.EndpointSet.get(_ENDPOINT_SET_RID)

    def v2_http_get() -> None:
        v2_http.admin.Enrollment.get(_ENROLLMENT_RID)

    def v1_http_get() -> None:
        v1_http.ontologies.Ontology.get(_ONTOLOGY_RID)

    def v2_stream() -> None:
        with v2_stream_client.admin.Enrollment.with_streaming_response.get(
            _ENROLLMENT_RID
        ) as response:
            for _chunk in response.iter_bytes():
                pass

    async def _async_v1() -> None:
        await async_v1.ontologies.Ontology.get(_ONTOLOGY_RID)

    async def _async_v2() -> None:
        await async_v2.admin.Enrollment.get(_ENROLLMENT_RID)

    async def _async_v3() -> None:
        await async_v3.endpoints.EndpointSet.get(_ENDPOINT_SET_RID)

    async def _async_v2_http() -> None:
        await async_v2_http.admin.Enrollment.get(_ENROLLMENT_RID)

    def token_fetch() -> None:
        confidential.get_token()

    def auth_code_exchange() -> None:
        public.set_token(code=_AUTH_CODE, state=public_state)

    def token_refresh() -> None:
        public_refresh.get_token()

    def token_revoke() -> None:
        confidential_revoke.sign_out()

    def lm_sync() -> None:
        sync_lm.get(f"http://{host}/stall")

    async def _lm_async() -> None:
        await async_lm.get(f"http://{host}/stall")

    return (
        ("v1 sync ontology get", v1_get, "connect"),
        ("v2 sync enrollment get", v2_get, "connect"),
        ("v3 sync endpoint set get", v3_get, "connect"),
        ("v1 async ontology get", lambda: asyncio.run(_async_v1()), "connect"),
        ("v2 async enrollment get", lambda: asyncio.run(_async_v2()), "connect"),
        ("v3 async endpoint set get", lambda: asyncio.run(_async_v3()), "connect"),
        ("confidential client token fetch", token_fetch, "connect"),
        ("public client authorization-code exchange", auth_code_exchange, "connect"),
        ("public client token refresh", token_refresh, "connect"),
        ("confidential client token revoke", token_revoke, "connect"),
        ("v1 sync ontology get over http", v1_http_get, "read"),
        ("v2 sync enrollment get over http", v2_http_get, "read"),
        ("v2 sync streaming read", v2_stream, "read"),
        ("v2 async enrollment get over http", lambda: asyncio.run(_async_v2_http()), "read"),
        ("language-model sync client get", lm_sync, "read"),
        ("language-model async client get", lambda: asyncio.run(_lm_async()), "read"),
    )
