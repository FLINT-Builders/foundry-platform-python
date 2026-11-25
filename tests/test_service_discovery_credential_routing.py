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


"""Service-discovery URLs must not receive Foundry credentials.

The documented stack address is FOUNDRY_HOSTNAME (or an explicit hostname
argument). When a client is built without an explicit hostname, URLs from
FOUNDRY_SERVICE_DISCOVERY_V2 are currently used as the request target.
These tests require that a discovery URL is not contacted when it is not
an https URL for the configured host with no userinfo. A client may refuse
the URL or ignore it. It must not open a connection to any other host, and
it must not send a credential there.

Resource methods other than the ones called here use the same
ApiClient._create_url and AsyncApiClient._create_url paths. Revoke for both
OAuth client types is covered through sign_out. PublicClientAuth.sign_in
builds an authorize URL and is checked below; it does not itself post a
token.
"""

import os
import ssl
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterator, List, Optional, Sequence
from urllib.parse import urlparse

import pytest

from foundry_sdk import (
    AsyncFoundryClient,
    ConfidentialClientAuth,
    Config,
    FoundryClient,
    PublicClientAuth,
)
from foundry_sdk._core.compute_module_pipeline_auth import ComputeModulePipelineAuth
from foundry_sdk._core.context_and_environment_vars import HOSTNAME_VAR, TOKEN_VAR
from foundry_sdk._core.oauth_utils import OAuthToken, OAuthTokenResponse
from foundry_sdk.v1 import FoundryClient as FoundryClientV1
from foundry_sdk.v3 import FoundryClient as FoundryClientV3

CONFIGURED_HOST = "foundry.invalid"
DATASET_RID = "ri.foundry.main.dataset.placeholder"
STREAM_BRANCH = "master"
PLACEHOLDER_TOKEN = "placeholder-token"
PLACEHOLDER_CLIENT_ID = "placeholder-client-id"
PLACEHOLDER_CLIENT_SECRET = "placeholder-client-secret"
PLACEHOLDER_REFRESH_TOKEN = "placeholder-refresh-token"
PLACEHOLDER_ACCESS_TOKEN = "placeholder-access-token"
PLACEHOLDER_AUTH_CODE = "placeholder-auth-code"
PLACEHOLDER_USERINFO_PASSWORD = "placeholder-userinfo-password"

NAMED_PLACEHOLDERS = (
    ("user token", PLACEHOLDER_TOKEN),
    ("client id", PLACEHOLDER_CLIENT_ID),
    ("client secret", PLACEHOLDER_CLIENT_SECRET),
    ("refresh token", PLACEHOLDER_REFRESH_TOKEN),
    ("access token", PLACEHOLDER_ACCESS_TOKEN),
    ("authorization code", PLACEHOLDER_AUTH_CODE),
    ("userinfo password", PLACEHOLDER_USERINFO_PASSWORD),
)


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._capture()

    def do_POST(self) -> None:
        self._capture()

    def _capture(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        self.server.records.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": body,
            }
        )
        payload = (
            b'{"access_token":"placeholder-access-token","token_type":"bearer",'
            b'"expires_in":3600,"refresh_token":"placeholder-refresh-token"}'
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class RecordingServer:
    def __init__(self, scheme: str) -> None:
        if scheme not in ("http", "https"):
            raise ValueError(scheme)
        self.scheme = scheme
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self._httpd.records = []  # type: ignore[attr-defined]
        if scheme == "https":
            self._tmpdir = tempfile.TemporaryDirectory()
            cert = os.path.join(self._tmpdir.name, "cert.pem")
            key = os.path.join(self._tmpdir.name, "key.pem")
            subprocess.check_call(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-keyout",
                    key,
                    "-out",
                    cert,
                    "-days",
                    "1",
                    "-nodes",
                    "-subj",
                    "/CN=127.0.0.1",
                    "-addext",
                    "subjectAltName=IP:127.0.0.1",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def origin(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.port}"

    @property
    def records(self) -> List[dict]:
        return self._httpd.records  # type: ignore[attr-defined]

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


@contextmanager
def recording_server(scheme: str) -> Iterator[RecordingServer]:
    server = RecordingServer(scheme)
    try:
        yield server
    finally:
        server.close()


def write_discovery(path, services: Dict[str, Sequence[str]]) -> None:
    lines: List[str] = []
    for name, urls in services.items():
        lines.append(f"{name}:")
        for url in urls:
            lines.append(f'  - "{url}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def client_config() -> Config:
    # verify is disabled so a local listener can finish the handshake and record
    # the request the SDK has already addressed. Destination selection does not
    # depend on certificate verification.
    return Config(verify=False, timeout=2)


def invoke(func):
    try:
        return func()
    except Exception:
        return None


def assert_reported_host(hostname: Optional[str], expected_host: str = CONFIGURED_HOST) -> None:
    if hostname is None:
        return
    parsed = urlparse(hostname)
    assert parsed.scheme == "https"
    assert parsed.hostname == expected_host
    assert parsed.username is None
    assert parsed.password is None


def assert_listener_idle(servers: Sequence[RecordingServer]) -> None:
    deliveries = []
    for server in servers:
        for record in server.records:
            carried = [
                name
                for name, value in NAMED_PLACEHOLDERS
                if value in record["headers"].get("authorization", "")
                or value.encode("utf-8") in record["body"]
            ]
            deliveries.append(
                f"{record['method']} {server.origin}{record['path']} carried {carried or 'headers or body'}"
            )
    assert not deliveries, (
        "service discovery URL was contacted; credentials must stay on the configured host: "
        + "; ".join(deliveries)
    )


@pytest.fixture
def discovery_file(monkeypatch, tmp_path):
    path = tmp_path / "services.yaml"
    monkeypatch.setenv("FOUNDRY_HOSTNAME", CONFIGURED_HOST)
    monkeypatch.setenv("FOUNDRY_SERVICE_DISCOVERY_V2", str(path))
    hostname_reset = HOSTNAME_VAR.set(None)
    token_reset = TOKEN_VAR.set(None)
    yield path
    HOSTNAME_VAR.reset(hostname_reset)
    TOKEN_VAR.reset(token_reset)


def _user_token_get(monkeypatch, client_factory, call) -> Optional[str]:
    monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
    config = client_config()
    client = invoke(lambda: client_factory(config))
    if client is None:
        return None
    reported = invoke(lambda: client.datasets._hostname)
    if reported is None and hasattr(client, "endpoints"):
        reported = client.endpoints._hostname
    assert_reported_host(reported)
    invoke(lambda: call(client))
    return reported


@pytest.mark.parametrize(
    "factory, call",
    [
        (
            lambda config: FoundryClient(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        ),
        (
            lambda config: FoundryClientV1(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        ),
        (
            lambda config: FoundryClientV3(config=config),
            lambda client: client.endpoints.EndpointSet.get(DATASET_RID),
        ),
    ],
    ids=["v2", "v1", "v3"],
)
def test_user_token_is_not_sent_to_https_discovery_api_gateway(
    monkeypatch, discovery_file, factory, call
):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"api-gateway": [server.origin]})
        _user_token_get(monkeypatch, factory, call)
        assert_listener_idle([server])


@pytest.mark.asyncio
async def test_async_user_token_is_not_sent_to_https_discovery_api_gateway(
    monkeypatch, discovery_file
):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"api-gateway": [server.origin]})
        monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
        config = client_config()
        client = invoke(lambda: AsyncFoundryClient(config=config, preview=True))
        if client is not None:
            assert_reported_host(client.datasets.Dataset._hostname)
            try:
                await client.datasets.Dataset.get(DATASET_RID)
            except Exception:
                pass
        assert_listener_idle([server])


def test_http_discovery_url_is_not_used_for_user_token(monkeypatch, discovery_file):
    with recording_server("http") as server:
        write_discovery(discovery_file, {"api-gateway": [server.origin]})
        _user_token_get(
            monkeypatch,
            lambda config: FoundryClient(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        )
        assert_listener_idle([server])


def test_userinfo_discovery_url_is_not_used_for_user_token(monkeypatch, discovery_file):
    with recording_server("http") as server:
        write_discovery(
            discovery_file,
            {
                "api-gateway": [
                    f"http://{CONFIGURED_HOST}:{PLACEHOLDER_USERINFO_PASSWORD}@{server.origin.split('://', 1)[1]}"
                ]
            },
        )
        _user_token_get(
            monkeypatch,
            lambda config: FoundryClient(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        )
        assert_listener_idle([server])


def test_https_userinfo_on_listener_host_is_not_used(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(
            discovery_file,
            {
                "api-gateway": [
                    f"https://placeholder-user:{PLACEHOLDER_USERINFO_PASSWORD}@{server.origin.split('://', 1)[1]}"
                ]
            },
        )
        _user_token_get(
            monkeypatch,
            lambda config: FoundryClient(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        )
        assert_listener_idle([server])


def test_http_url_for_configured_listener_host_is_not_used(monkeypatch, discovery_file):
    """An http URL whose host equals FOUNDRY_HOSTNAME is still not a safe target."""
    with recording_server("http") as server:
        monkeypatch.setenv("FOUNDRY_HOSTNAME", "127.0.0.1")
        write_discovery(discovery_file, {"api-gateway": [server.origin]})
        monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
        config = client_config()
        client = invoke(lambda: FoundryClient(config=config))
        if client is not None:
            assert_reported_host(client.datasets._hostname, expected_host="127.0.0.1")
            invoke(lambda: client.datasets.Dataset.get(DATASET_RID))
        assert_listener_idle([server])


def test_underscore_service_name_is_not_used_for_user_token(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"api_gateway": [server.origin]})
        _user_token_get(
            monkeypatch,
            lambda config: FoundryClient(config=config),
            lambda client: client.datasets.Dataset.get(DATASET_RID),
        )
        assert_listener_idle([server])


def test_high_scale_bearer_is_not_sent_to_stream_proxy_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"stream_proxy": [server.origin]})
        monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
        config = client_config()
        client = invoke(lambda: FoundryClient(config=config))
        if client is not None:
            assert_reported_host(client.streams._hostname)
            invoke(
                lambda: client.streams.Dataset.Stream.get_end_offsets(
                    DATASET_RID, STREAM_BRANCH, preview=True
                )
            )
        assert_listener_idle([server])


@pytest.mark.asyncio
async def test_async_high_scale_bearer_is_not_sent_to_stream_proxy_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"stream-proxy": [server.origin]})
        monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
        config = client_config()
        client = invoke(lambda: AsyncFoundryClient(config=config, preview=True))
        if client is not None:
            assert_reported_host(client.streams.Dataset._hostname)
            try:
                await client.streams.Dataset.Stream.get_end_offsets(
                    DATASET_RID, STREAM_BRANCH, preview=True
                )
            except Exception:
                pass
        assert_listener_idle([server])


def test_discovery_file_replaced_before_the_call_is_not_followed(monkeypatch, discovery_file):
    with recording_server("http") as first, recording_server("http") as second:
        write_discovery(discovery_file, {"api-gateway": [first.origin]})
        monkeypatch.setenv("FOUNDRY_TOKEN", PLACEHOLDER_TOKEN)
        config = client_config()
        client = invoke(lambda: FoundryClient(config=config))
        write_discovery(discovery_file, {"api-gateway": [second.origin]})
        if client is not None:
            assert_reported_host(client.datasets._hostname)
            invoke(lambda: client.datasets.Dataset.get(DATASET_RID))
        assert_listener_idle([first, second])


def test_confidential_client_secret_is_not_posted_to_multipass_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(
            discovery_file,
            {"api-gateway": [server.origin], "multipass": [server.origin + "/multipass"]},
        )
        config = client_config()
        auth = ConfidentialClientAuth(
            client_id=PLACEHOLDER_CLIENT_ID,
            client_secret=PLACEHOLDER_CLIENT_SECRET,
            should_refresh=False,
            config=config,
        )
        client = invoke(lambda: FoundryClient(auth=auth, config=config))
        if client is not None:
            assert_reported_host(client.datasets._hostname)
            invoke(lambda: client.datasets.Dataset.get(DATASET_RID))
        assert_listener_idle([server])


def test_confidential_revoke_is_not_posted_to_multipass_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"multipass": [server.origin]})
        config = client_config()
        auth = ConfidentialClientAuth(
            client_id=PLACEHOLDER_CLIENT_ID,
            client_secret=PLACEHOLDER_CLIENT_SECRET,
            should_refresh=False,
            config=config,
        )
        client = invoke(lambda: FoundryClient(auth=auth, config=config))
        if client is not None:
            _ = client.datasets.Dataset
            auth._token = OAuthToken(
                token=OAuthTokenResponse(
                    {
                        "access_token": PLACEHOLDER_ACCESS_TOKEN,
                        "token_type": "bearer",
                        "expires_in": 3600,
                    }
                )
            )
            invoke(auth.sign_out)
        assert_listener_idle([server])


def test_public_refresh_token_is_not_posted_to_multipass_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"multipass": [server.origin]})
        config = client_config()
        auth = PublicClientAuth(
            client_id=PLACEHOLDER_CLIENT_ID,
            redirect_url="https://foundry.invalid/callback",
            should_refresh=False,
            config=config,
        )
        client = invoke(lambda: FoundryClient(auth=auth, config=config))
        if client is not None:
            _ = client.datasets.Dataset
            auth._token = OAuthToken(
                token=OAuthTokenResponse(
                    {
                        "access_token": PLACEHOLDER_ACCESS_TOKEN,
                        "token_type": "bearer",
                        "expires_in": 0,
                        "refresh_token": PLACEHOLDER_REFRESH_TOKEN,
                    }
                )
            )
            invoke(auth.get_token)
        assert_listener_idle([server])


def test_public_authorization_code_is_not_posted_to_multipass_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"multipass": [server.origin]})
        config = client_config()
        auth = PublicClientAuth(
            client_id=PLACEHOLDER_CLIENT_ID,
            redirect_url="https://foundry.invalid/callback",
            should_refresh=False,
            config=config,
        )
        client = invoke(lambda: FoundryClient(auth=auth, config=config))
        url = None
        if client is not None:
            _ = client.datasets.Dataset
            url = invoke(auth.sign_in)
            if auth._auth_request is not None:
                invoke(
                    lambda: auth.set_token(
                        code=PLACEHOLDER_AUTH_CODE, state=auth._auth_request.state
                    )
                )
        assert_listener_idle([server])
        if url is not None:
            parsed = urlparse(url)
            assert parsed.hostname == CONFIGURED_HOST
            assert parsed.scheme == "https"
            assert parsed.username is None
            assert parsed.password is None


def test_public_revoke_is_not_posted_to_multipass_url(monkeypatch, discovery_file):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"multipass": [server.origin]})
        config = client_config()
        auth = PublicClientAuth(
            client_id=PLACEHOLDER_CLIENT_ID,
            redirect_url="https://foundry.invalid/callback",
            should_refresh=False,
            config=config,
        )
        client = invoke(lambda: FoundryClient(auth=auth, config=config))
        if client is not None:
            _ = client.datasets.Dataset
            auth._token = OAuthToken(
                token=OAuthTokenResponse(
                    {
                        "access_token": PLACEHOLDER_ACCESS_TOKEN,
                        "token_type": "bearer",
                        "expires_in": 3600,
                    }
                )
            )
            invoke(auth.sign_out)
        assert_listener_idle([server])


def test_compute_module_token_is_not_sent_to_discovery_api_gateway(
    monkeypatch, discovery_file, tmp_path
):
    with recording_server("https") as server:
        write_discovery(discovery_file, {"api-gateway": [server.origin]})
        token_file = tmp_path / "pipeline-token"
        token_file.write_text(PLACEHOLDER_TOKEN + "\n", encoding="utf-8")
        monkeypatch.setenv("BUILD2_TOKEN", str(token_file))
        config = client_config()
        client = invoke(lambda: FoundryClient(auth=ComputeModulePipelineAuth(), config=config))
        if client is not None:
            assert_reported_host(client.datasets._hostname)
            invoke(lambda: client.datasets.Dataset.get(DATASET_RID))
        assert_listener_idle([server])
