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


"""Path parameters of "." and ".." must not change the endpoint httpx sends.

These tests fail while those values are placed into the URL with the dots left
intact. httpx then removes the dot segment, and the request is delivered to a
different API path. A fix has to reject the value before any request is sent,
or percent-encode the dots so the segment is still present on the wire.

Other generated resources are not called one by one. They all substitute path
parameters in BaseApiClient._create_url, which the shared cases below call
directly. Values that are not a whole dot segment, such as "1.0" and "a/..",
must still be sent.
"""

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest

from foundry_sdk._core.api_client import ApiClient, AsyncApiClient, RequestInfo
from foundry_sdk._core.config import Config
from foundry_sdk._core.resource_iterator import ResourceIterator
from foundry_sdk._core.user_token_auth_client import UserTokenAuth
from foundry_sdk.v1.ontologies.ontology_object import (
    OntologyObjectClient as V1OntologyObjectClient,
)
from foundry_sdk.v2.datasets.branch import BranchClient
from foundry_sdk.v2.datasets.file import FileClient
from foundry_sdk.v2.ontologies.linked_object import LinkedObjectClient
from foundry_sdk.v2.ontologies.ontology_object import AsyncOntologyObjectClient
from foundry_sdk.v2.ontologies.ontology_object import (
    OntologyObjectClient as V2OntologyObjectClient,
)

HOST = "example.test"
# Placeholder credential. The tests never send it to a network service.
AUTH = UserTokenAuth(token="placeholder-token")
CONFIG = Config(scheme="https")
DATASET = "ri.foundry.main.dataset.abc123"
ONTOLOGY_RID = "ri.ontology.main.ontology.abc"
TX_BRANCH = "ri.foundry.main.transaction.branchonly"
TX_DATASET = "ri.foundry.main.transaction.datasetwide"

SINGLE_OBJECT = {"orderId": "only-one"}
OBJECT_LIST_PAGE = {
    "data": [{"orderId": "secret-1"}, {"orderId": "secret-2"}],
    "nextPageToken": None,
    "totalCount": "2",
}
V1_OBJECT = {
    "properties": {"orderId": "only-one"},
    "rid": "ri.phonograph.main.object.1",
}
FILE_METADATA = {
    "path": "report.txt",
    "transactionRid": TX_BRANCH,
    "updatedTime": "2021-01-04T05:00:00Z",
}
BRANCH_TX_PAGE = {
    "data": [
        {
            "rid": TX_BRANCH,
            "transactionType": "UPDATE",
            "status": "COMMITTED",
            "createdTime": "2021-01-04T05:00:00Z",
        }
    ],
    "nextPageToken": None,
}
DATASET_TX_PAGE = {
    "data": [
        {
            "rid": TX_DATASET,
            "transactionType": "SNAPSHOT",
            "status": "COMMITTED",
            "createdTime": "2021-01-04T05:00:00Z",
        }
    ],
    "nextPageToken": None,
}

Sent = Tuple[str, str]
CaseCall = Callable[[Any], Any]


def _api(*segments: str) -> str:
    return "/" + "/".join(("api", *segments))


def _canonical_segment(segment: str) -> str:
    """Decode %2E, including a %25 that httpx may add when it quotes the path again."""
    previous = None
    current = segment
    while current != previous:
        previous = current
        current = re.sub("%25", "%", current, flags=re.IGNORECASE)
        current = re.sub("%2[Ee]", ".", current)
    return current


def _canonical(path: str) -> str:
    return "/".join(_canonical_segment(segment) for segment in path.split("/"))


def _describe(result: Any) -> str:
    if isinstance(result, ResourceIterator):
        try:
            return repr(list(result))
        except Exception as exc:
            return f"{result!r} (iteration failed: {exc})"
    return repr(result)


class _Capture:
    def __init__(self) -> None:
        self.sent: List[Sent] = []
        self.fallback: Any = SINGLE_OBJECT
        self.by_path: Dict[str, Any] = {}

    def install(self, client: Any) -> None:
        api = client if isinstance(client, (ApiClient, AsyncApiClient)) else client._api_client
        transport = httpx.MockTransport(self.handler)
        if isinstance(api, AsyncApiClient):
            api._client._transport = transport
        else:
            api._session._transport = transport

    def handler(self, request: httpx.Request) -> httpx.Response:
        raw = request.url.raw_path.split(b"?", 1)[0].decode("ascii")
        self.sent.append((request.method, raw))
        body = self.by_path.get(raw, self.fallback)
        if isinstance(body, (bytes, bytearray)):
            return httpx.Response(200, content=bytes(body))
        return httpx.Response(200, json=body)


def _client(kind: str) -> Any:
    if kind == "api":
        return ApiClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "v2-object":
        return V2OntologyObjectClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "v2-object-async":
        return AsyncOntologyObjectClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "v2-linked":
        return LinkedObjectClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "v1-object":
        return V1OntologyObjectClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "branch":
        return BranchClient(auth=AUTH, hostname=HOST, config=CONFIG)
    if kind == "file":
        return FileClient(auth=AUTH, hostname=HOST, config=CONFIG)
    raise AssertionError(kind)


def _assert_endpoint(
    capture: _Capture,
    kind: str,
    call: CaseCall,
    method: str,
    expected_path: str,
    check: Optional[str],
) -> None:
    capture.sent.clear()
    client = _client(kind)
    capture.install(client)
    try:
        result = call(client)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as exc:
        if not capture.sent:
            return
        wanted = [(method, _canonical(expected_path))]
        actual = [(sent_method, _canonical(path)) for sent_method, path in capture.sent]
        if actual != wanted:
            raise AssertionError(
                "dot segment changed the endpoint before the call failed: "
                f"sent {capture.sent}, expected {wanted}"
            ) from exc
        raise
    wanted = [(method, _canonical(expected_path))]
    actual = [(sent_method, _canonical(path)) for sent_method, path in capture.sent]
    assert actual == wanted, (
        f"dot segment changed the endpoint: sent {capture.sent}, expected {wanted}, "
        f"result={_describe(result)}"
    )
    if check == "single":
        assert result == SINGLE_OBJECT
    elif check == "branch-tx":
        assert [row.rid for row in result] == [TX_BRANCH]
    elif check == "no-listing":
        assert result != OBJECT_LIST_PAGE
        assert "secret-1" not in repr(result)


@pytest.fixture
def capture() -> _Capture:
    return _Capture()


# kind, http method, expected wire path, fallback body, collapsed bodies, call, check
_CASES = [
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "."),
        SINGLE_OBJECT,
        {_api("v2", "ontologies", "ont", "objects", "Order"): OBJECT_LIST_PAGE},
        lambda c: c.get("ont", "Order", "."),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", ".."),
        SINGLE_OBJECT,
        {_api("v2", "ontologies", "ont", "objects"): OBJECT_LIST_PAGE},
        lambda c: c.get("ont", "Order", ".."),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", ".", "order-1"),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", ".", "order-1"),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", ".", ".."),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", ".", ".."),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "order-1"),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", "Order", "order-1"),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "1.0"),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", "Order", "1.0"),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "a%2F.."),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", "Order", "a/.."),
        "single",
    ),
    (
        "v2-object",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", ""),
        SINGLE_OBJECT,
        {},
        lambda c: c.get("ont", "Order", ""),
        "no-listing",
    ),
    (
        "v2-object-async",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "."),
        SINGLE_OBJECT,
        {_api("v2", "ontologies", "ont", "objects", "Order"): OBJECT_LIST_PAGE},
        lambda c: c.get("ont", "Order", "."),
        "single",
    ),
    (
        "v2-linked",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "order-1", "links", "placed", "."),
        SINGLE_OBJECT,
        {
            _api(
                "v2", "ontologies", "ont", "objects", "Order", "order-1", "links", "placed"
            ): OBJECT_LIST_PAGE
        },
        lambda c: c.get_linked_object("ont", "Order", "order-1", "placed", "."),
        "single",
    ),
    (
        "v2-linked",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", "order-1", "links", "placed", ".."),
        SINGLE_OBJECT,
        {},
        lambda c: c.get_linked_object("ont", "Order", "order-1", "placed", ".."),
        "single",
    ),
    (
        "v2-linked",
        "GET",
        _api("v2", "ontologies", "ont", "objects", "Order", ".", "links", "placed", "c-1"),
        SINGLE_OBJECT,
        {},
        lambda c: c.get_linked_object("ont", "Order", ".", "placed", "c-1"),
        "single",
    ),
    (
        "v1-object",
        "GET",
        _api("v1", "ontologies", ONTOLOGY_RID, "objects", "Order", "."),
        V1_OBJECT,
        {},
        lambda c: c.get(ONTOLOGY_RID, "Order", "."),
        None,
    ),
    (
        "v1-object",
        "GET",
        _api("v1", "ontologies", ONTOLOGY_RID, "objects", "Order", ".."),
        V1_OBJECT,
        {},
        lambda c: c.get(ONTOLOGY_RID, "Order", ".."),
        None,
    ),
    (
        "v1-object",
        "GET",
        _api(
            "v1",
            "ontologies",
            ONTOLOGY_RID,
            "objects",
            "Order",
            "order-1",
            "links",
            "placed",
            ".",
        ),
        V1_OBJECT,
        {},
        lambda c: c.get_linked_object(ONTOLOGY_RID, "Order", "order-1", "placed", "."),
        None,
    ),
    (
        "branch",
        "GET",
        _api("v2", "datasets", DATASET, "branches", "..", "transactions"),
        BRANCH_TX_PAGE,
        {_api("v2", "datasets", DATASET, "transactions"): DATASET_TX_PAGE},
        lambda c: c.transactions(DATASET, ".."),
        "branch-tx",
    ),
    (
        "branch",
        "GET",
        _api("v2", "datasets", DATASET, "branches", ".", "transactions"),
        BRANCH_TX_PAGE,
        {},
        lambda c: c.transactions(DATASET, "."),
        "branch-tx",
    ),
    (
        "branch",
        "GET",
        _api("v2", "datasets", DATASET, "branches", "."),
        {"name": "kept"},
        {},
        lambda c: c.get(DATASET, "."),
        None,
    ),
    (
        "branch",
        "GET",
        _api("v2", "datasets", DATASET, "branches", ".."),
        {"name": "kept"},
        {},
        lambda c: c.get(DATASET, ".."),
        None,
    ),
    (
        "branch",
        "DELETE",
        _api("v2", "datasets", DATASET, "branches", "."),
        {},
        {},
        lambda c: c.delete(DATASET, "."),
        None,
    ),
    (
        "branch",
        "DELETE",
        _api("v2", "datasets", DATASET, "branches", ".."),
        {},
        {},
        lambda c: c.delete(DATASET, ".."),
        None,
    ),
    (
        "file",
        "GET",
        _api("v2", "datasets", DATASET, "files", "."),
        FILE_METADATA,
        {_api("v2", "datasets", DATASET, "files"): OBJECT_LIST_PAGE},
        lambda c: c.get(DATASET, "."),
        None,
    ),
    (
        "file",
        "GET",
        _api("v2", "datasets", DATASET, "files", ".."),
        FILE_METADATA,
        {},
        lambda c: c.get(DATASET, ".."),
        None,
    ),
    (
        "file",
        "GET",
        _api("v2", "datasets", DATASET, "files", ".", "content"),
        b"file-bytes",
        {},
        lambda c: c.content(DATASET, "."),
        None,
    ),
    (
        "file",
        "GET",
        _api("v2", "datasets", DATASET, "files", "..", "content"),
        b"file-bytes",
        {},
        lambda c: c.content(DATASET, ".."),
        None,
    ),
    (
        "file",
        "DELETE",
        _api("v2", "datasets", DATASET, "files", "."),
        {},
        {},
        lambda c: c.delete(DATASET, "."),
        None,
    ),
    (
        "file",
        "DELETE",
        _api("v2", "datasets", DATASET, "files", ".."),
        {},
        {},
        lambda c: c.delete(DATASET, ".."),
        None,
    ),
    (
        "file",
        "POST",
        _api("v2", "datasets", DATASET, "files", ".", "upload"),
        FILE_METADATA,
        {},
        lambda c: c.upload(DATASET, ".", b"placeholder-bytes"),
        None,
    ),
    (
        "file",
        "POST",
        _api("v2", "datasets", DATASET, "files", "..", "upload"),
        FILE_METADATA,
        {},
        lambda c: c.upload(DATASET, "..", b"placeholder-bytes"),
        None,
    ),
    (
        "api",
        "GET",
        _api("v2", "widgets", "."),
        {"ok": True},
        {},
        lambda c: c.call_api(
            RequestInfo.with_defaults(
                method="GET",
                resource_path="/v2/widgets/{widget}",
                path_params={"widget": "."},
                response_type=dict,
            )
        ),
        None,
    ),
    (
        "api",
        "GET",
        _api("v2", "widgets", "..", "items", "kept"),
        {"ok": True},
        {},
        lambda c: c.call_api(
            RequestInfo.with_defaults(
                method="GET",
                resource_path="/v2/widgets/{widget}/items/{item}",
                path_params={"widget": "..", "item": "kept"},
                response_type=dict,
            )
        ),
        None,
    ),
    (
        "api",
        "GET",
        _api("v2", "files", "..", "content"),
        {"ok": True},
        {},
        lambda c: c.call_api(
            RequestInfo.with_defaults(
                method="GET",
                resource_path="/v2/files/{filePath}/content",
                path_params={"filePath": ".."},
                response_type=dict,
            )
        ),
        None,
    ),
]


def _case_id(index: int, case: tuple) -> str:
    kind, method, path = case[0], case[1], case[2]
    return f"{index}-{kind}-{method.lower()}-{path.strip('/').replace('/', '-')}"


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[_case_id(index, case) for index, case in enumerate(_CASES)],
)
def test_dot_segment_path_parameter_keeps_its_endpoint(capture: _Capture, case: tuple) -> None:
    kind, method, expected, fallback, by_path, call, check = case
    capture.fallback = fallback
    capture.by_path = by_path
    _assert_endpoint(capture, kind, call, method, expected, check)
