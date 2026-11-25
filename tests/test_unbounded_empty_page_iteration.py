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


"""Bounds for paged list calls that receive empty pages with a next-page token.

Every sync list method with response mode ITERATOR uses ``ApiClient._call_api``
and ``ResourceIterator``. Every async list method uses ``AsyncApiClient._call_api``
and ``AsyncResourceIterator``. These tests drive that shared path through the
public v1, v2, and v3 clients (dataset files, audit log files, admin users, and
endpoint-set endpoints). Other list methods are the same two branches with a
different response model; they are not called individually.

Contract: one iteration step, including the page fetched while a sync iterator
is constructed, may perform at most ``MAX_EMPTY_PAGE_REQUESTS`` HTTP requests
while every page has an empty ``data`` array and a ``nextPageToken``. Past that
limit the step must raise, must not yield a later row, and must not raise
``RecursionError``. Pages that contain rows, and a final empty page with no
token, must still iterate.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any, Optional

import httpx
import pytest

import foundry_sdk._core.http_client as http_client
from foundry_sdk import Config, UserTokenAuth
from foundry_sdk.v1 import AsyncFoundryClient as AsyncFoundryClientV1
from foundry_sdk.v1 import FoundryClient as FoundryClientV1
from foundry_sdk.v2 import AsyncFoundryClient as AsyncFoundryClientV2
from foundry_sdk.v2 import FoundryClient as FoundryClientV2
from foundry_sdk.v3 import AsyncFoundryClient as AsyncFoundryClientV3
from foundry_sdk.v3 import FoundryClient as FoundryClientV3

# One iteration step must not keep requesting once this many pages in a row
# were empty but still carried a next-page token.
MAX_EMPTY_PAGE_REQUESTS = 16

# Longer than the cap, short of the default call-stack limit.
EMPTY_PAGES_BEFORE_ITEM = 40

# Longer than the default recursion limit, so an async implementation that
# recurses once per empty page overflows before it can reach a later row.
EMPTY_PAGES_THAT_OVERFLOW_THE_STACK = 1200

PLACEHOLDER_TOKEN = "placeholder-token"
HOSTNAME = "foundry.invalid"
DATASET_RID = "ri.foundry.main.dataset.placeholder"
ORGANIZATION_RID = "ri.foundry.main.organization.placeholder"
ENDPOINT_SET_RID = "ri.foundry.main.endpoint-set.placeholder"

FILE_ITEM = {
    "path": "placeholder.txt",
    "transactionRid": "ri.foundry.main.transaction.placeholder",
    "updatedTime": "2024-01-01T00:00:00+00:00",
}
LOG_FILE_ITEM = {"id": "placeholder-log"}
USER_ITEM = {
    "id": "00000000-0000-0000-0000-000000000001",
    "username": "placeholder",
    "realm": "placeholder",
    "status": "ACTIVE",
    "attributes": {},
}
ENDPOINT_ITEM = {
    "name": "placeholder",
    "rid": "ri.foundry.main.endpoint.placeholder",
}


def _empty_page(token: str) -> dict:
    return {"data": [], "nextPageToken": token}


def _item_page(item: dict, token: Optional[str]) -> dict:
    return {"data": [item], "nextPageToken": token}


def _empty_run_then_item(count: int, item: dict, token_for_index: Callable[[int], str]) -> list:
    pages = [_empty_page(token_for_index(index)) for index in range(1, count + 1)]
    pages.append(_item_page(item, None))
    return pages


class _ScriptedPages:
    def __init__(self, pages: list) -> None:
        self.pages = pages
        self.requests: list[httpx.Request] = []

    def response_for(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.pages):
            raise AssertionError(
                f"list iteration issued request {index + 1}, "
                f"past the {len(self.pages)} scripted pages"
            )
        return httpx.Response(200, json=self.pages[index])

    @property
    def count(self) -> int:
        return len(self.requests)


class _SyncTransport(httpx.BaseTransport):
    def __init__(self, script: _ScriptedPages) -> None:
        self._script = script

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._script.response_for(request)


class _AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, script: _ScriptedPages) -> None:
        self._script = script

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._script.response_for(request)


@contextmanager
def _scripted_http(pages: list) -> Iterator[_ScriptedPages]:
    script = _ScriptedPages(pages)
    original_sync = http_client._get_transport
    original_async = http_client._get_async_transport
    http_client._get_transport = lambda verify, proxy_url: _SyncTransport(script)
    http_client._get_async_transport = lambda verify, proxy_url: _AsyncTransport(script)
    try:
        yield script
    finally:
        http_client._get_transport = original_sync
        http_client._get_async_transport = original_async


def _config() -> Config:
    return Config(scheme="http")


def _auth() -> UserTokenAuth:
    return UserTokenAuth(token=PLACEHOLDER_TOKEN)


def _sync_client(version: str):
    if version == "v1":
        return FoundryClientV1(auth=_auth(), hostname=HOSTNAME, config=_config())
    if version == "v2":
        return FoundryClientV2(auth=_auth(), hostname=HOSTNAME, config=_config())
    if version == "v3":
        return FoundryClientV3(auth=_auth(), hostname=HOSTNAME, config=_config())
    raise AssertionError(version)


def _async_client(version: str):
    if version == "v1":
        return AsyncFoundryClientV1(
            auth=_auth(), hostname=HOSTNAME, config=_config(), preview=True
        )
    if version == "v2":
        return AsyncFoundryClientV2(
            auth=_auth(), hostname=HOSTNAME, config=_config(), preview=True
        )
    if version == "v3":
        return AsyncFoundryClientV3(
            auth=_auth(), hostname=HOSTNAME, config=_config(), preview=True
        )
    raise AssertionError(version)


def _list_v2_files(client):
    return client.datasets.Dataset.File.list(DATASET_RID)


def _list_v1_files(client):
    return client.datasets.Dataset.File.list(DATASET_RID)


def _list_audit_logs(client):
    return client.audit.Organization.LogFile.list(ORGANIZATION_RID, start_date=date(2024, 1, 1))


def _list_admin_users(client):
    return client.admin.User.list()


def _list_endpoints(client):
    return client.endpoints.EndpointSet.Endpoint.list(ENDPOINT_SET_RID)


# Public list operations that share the iterator. Each uses a different response model.
SYNC_LIST_CASES = [
    ("v2-dataset-files", "v2", _list_v2_files, FILE_ITEM, f"/v2/datasets/{DATASET_RID}/files"),
    (
        "v2-audit-log-files",
        "v2",
        _list_audit_logs,
        LOG_FILE_ITEM,
        f"/v2/audit/organizations/{ORGANIZATION_RID}/logFiles",
    ),
    ("v2-admin-users", "v2", _list_admin_users, USER_ITEM, "/v2/admin/users"),
    ("v1-dataset-files", "v1", _list_v1_files, FILE_ITEM, f"/v1/datasets/{DATASET_RID}/files"),
    (
        "v3-endpoint-set-endpoints",
        "v3",
        _list_endpoints,
        ENDPOINT_ITEM,
        f"/v3/platform/endpointSets/{ENDPOINT_SET_RID}/endpoints",
    ),
]

ASYNC_LIST_CASES = [
    ("v2-dataset-files", "v2", _list_v2_files, FILE_ITEM, f"/v2/datasets/{DATASET_RID}/files"),
    (
        "v2-audit-log-files",
        "v2",
        _list_audit_logs,
        LOG_FILE_ITEM,
        f"/v2/audit/organizations/{ORGANIZATION_RID}/logFiles",
    ),
    ("v2-admin-users", "v2", _list_admin_users, USER_ITEM, "/v2/admin/users"),
    ("v1-dataset-files", "v1", _list_v1_files, FILE_ITEM, f"/v1/datasets/{DATASET_RID}/files"),
    (
        "v3-endpoint-set-endpoints",
        "v3",
        _list_endpoints,
        ENDPOINT_ITEM,
        f"/v3/platform/endpointSets/{ENDPOINT_SET_RID}/endpoints",
    ),
]


def _assert_requests_follow_the_script(script: _ScriptedPages, path: str) -> None:
    assert script.count >= 1
    for index, request in enumerate(script.requests):
        assert request.method == "GET"
        assert request.url.host == HOSTNAME
        assert path in str(request.url)
        assert request.headers["authorization"] == "Bearer " + PLACEHOLDER_TOKEN
        if index == 0:
            continue
        previous_token = script.pages[index - 1]["nextPageToken"]
        assert request.url.params.get("pageToken") == previous_token


def _assert_step_stopped_within_bound(
    item: Any, error: Optional[BaseException], script: _ScriptedPages
) -> None:
    assert not isinstance(error, RecursionError), (
        f"one iteration step raised RecursionError after {script.count} empty pages"
    )
    assert error is not None, (
        f"one iteration step yielded {item!r} after {script.count} requests; "
        f"empty pages that still include a next-page token must stop within "
        f"{MAX_EMPTY_PAGE_REQUESTS} requests and must not yield a later row"
    )
    assert script.count <= MAX_EMPTY_PAGE_REQUESTS, (
        f"one iteration step issued {script.count} requests for consecutive empty pages; "
        f"the limit is {MAX_EMPTY_PAGE_REQUESTS}"
    )


def _first_sync(iterator):
    try:
        return next(iterator), None
    except Exception as exc:
        return None, exc


async def _first_async(iterator):
    try:
        return await anext(iterator), None
    except Exception as exc:
        return None, exc


@pytest.mark.parametrize(
    ("name", "version", "start", "item", "path"),
    SYNC_LIST_CASES,
    ids=[case[0] for case in SYNC_LIST_CASES],
)
def test_sync_empty_pages_with_distinct_tokens_are_bounded(name, version, start, item, path):
    pages = _empty_run_then_item(EMPTY_PAGES_BEFORE_ITEM, item, lambda index: f"empty-{index}")
    with _scripted_http(pages) as script:
        iterator = start(_sync_client(version))
        found, error = _first_sync(iterator)
        _assert_requests_follow_the_script(script, path)
        _assert_step_stopped_within_bound(found, error, script)


def test_sync_repeated_page_token_is_bounded():
    pages = _empty_run_then_item(EMPTY_PAGES_BEFORE_ITEM, FILE_ITEM, lambda index: "same-token")
    with _scripted_http(pages) as script:
        iterator = _list_v2_files(_sync_client("v2"))
        found, error = _first_sync(iterator)
        _assert_requests_follow_the_script(script, f"/v2/datasets/{DATASET_RID}/files")
        _assert_step_stopped_within_bound(found, error, script)


@pytest.mark.parametrize(
    ("name", "version", "start", "item", "path"),
    ASYNC_LIST_CASES,
    ids=[case[0] for case in ASYNC_LIST_CASES],
)
@pytest.mark.asyncio
async def test_async_empty_pages_with_distinct_tokens_are_bounded(name, version, start, item, path):
    pages = _empty_run_then_item(EMPTY_PAGES_BEFORE_ITEM, item, lambda index: f"empty-{index}")
    with _scripted_http(pages) as script:
        iterator = start(_async_client(version))
        found, error = await _first_async(iterator)
        _assert_requests_follow_the_script(script, path)
        _assert_step_stopped_within_bound(found, error, script)


@pytest.mark.asyncio
async def test_async_empty_pages_do_not_overflow_the_call_stack():
    pages = _empty_run_then_item(
        EMPTY_PAGES_THAT_OVERFLOW_THE_STACK,
        FILE_ITEM,
        lambda index: f"empty-{index}",
    )
    with _scripted_http(pages) as script:
        iterator = _list_v2_files(_async_client("v2"))
        found, error = await _first_async(iterator)
        _assert_requests_follow_the_script(script, f"/v2/datasets/{DATASET_RID}/files")
        _assert_step_stopped_within_bound(found, error, script)


@pytest.mark.parametrize(
    ("name", "version", "start", "item", "path"),
    [SYNC_LIST_CASES[0], SYNC_LIST_CASES[1]],
    ids=["v2-dataset-files", "v2-audit-log-files"],
)
def test_sync_nonempty_pages_still_yield_every_item(name, version, start, item, path):
    pages = [
        _item_page(item, "page-2"),
        _item_page(item, None),
    ]
    with _scripted_http(pages) as script:
        iterator = start(_sync_client(version))
        found = list(iterator)
        _assert_requests_follow_the_script(script, path)
        assert len(found) == 2
        assert script.count == 2


def test_sync_final_empty_page_without_token_ends_iteration():
    pages = [
        _item_page(FILE_ITEM, "page-2"),
        _empty_page_without_token(),
    ]
    with _scripted_http(pages) as script:
        iterator = _list_v2_files(_sync_client("v2"))
        found = list(iterator)
        _assert_requests_follow_the_script(script, f"/v2/datasets/{DATASET_RID}/files")
        assert len(found) == 1
        assert script.count == 2


@pytest.mark.asyncio
async def test_async_nonempty_pages_still_yield_every_item():
    pages = [
        _item_page(FILE_ITEM, "page-2"),
        _item_page(FILE_ITEM, None),
    ]
    with _scripted_http(pages) as script:
        iterator = _list_v2_files(_async_client("v2"))
        found = [item async for item in iterator]
        _assert_requests_follow_the_script(script, f"/v2/datasets/{DATASET_RID}/files")
        assert len(found) == 2
        assert script.count == 2


@pytest.mark.asyncio
async def test_async_final_empty_page_without_token_ends_iteration():
    pages = [
        _item_page(FILE_ITEM, "page-2"),
        _empty_page_without_token(),
    ]
    with _scripted_http(pages) as script:
        iterator = _list_v2_files(_async_client("v2"))
        found = [item async for item in iterator]
        _assert_requests_follow_the_script(script, f"/v2/datasets/{DATASET_RID}/files")
        assert len(found) == 1
        assert script.count == 2


def _empty_page_without_token() -> dict:
    return {"data": [], "nextPageToken": None}
