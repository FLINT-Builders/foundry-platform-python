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


from typing import Generic, List, NoReturn, Optional, TypeVar

from foundry_sdk._core.page_iterator import (
    AsyncPageFunction,
    AsyncPageIterator,
    PageFunction,
    PageIterator,
)
from foundry_sdk._errors import PaginationError

T = TypeVar("T")

# A server that keeps answering with empty pages and a next page token would otherwise make a
# single iteration step issue requests without end, since no item is ever handed to the caller.
MAX_CONSECUTIVE_EMPTY_PAGES = 16


def _raise_too_many_empty_pages() -> NoReturn:
    raise PaginationError(
        f"Stopped paging after {MAX_CONSECUTIVE_EMPTY_PAGES} consecutive empty pages that each "
        "returned a next page token."
    )


class ResourceIterator(Generic[T]):
    """A generic class for iterating over paged responses."""

    def __init__(
        self,
        paged_func: PageFunction[T],
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> None:
        self._page_iterator = PageIterator(paged_func, page_size, page_token)
        self._index = 0
        self._consecutive_empty_pages = 0
        self._record_page(self._page_iterator.data)

    @property
    def data(self) -> List[T]:
        return self._page_iterator.data

    @property
    def next_page_token(self) -> Optional[str]:
        return self._page_iterator.next_page_token

    def __iter__(self):
        return self

    def __next__(self):
        while self._index >= len(self._page_iterator.data):
            self._get_data()

        obj = self._page_iterator.data[self._index]
        self._index += 1
        return obj

    def _get_data(self):
        if (
            self._consecutive_empty_pages >= MAX_CONSECUTIVE_EMPTY_PAGES
            and self._page_iterator.next_page_token is not None
        ):
            _raise_too_many_empty_pages()

        try:
            next(self._page_iterator)
            self._index = 0
        except StopIteration:
            raise StopIteration("End of iteration reached")

        self._record_page(self._page_iterator.data)

    def _record_page(self, data: List[T]) -> None:
        self._consecutive_empty_pages = 0 if data else self._consecutive_empty_pages + 1


class AsyncResourceIterator(Generic[T]):
    """A generic class for async iterating over paged responses."""

    def __init__(
        self,
        paged_func: AsyncPageFunction[T],
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> None:
        self._page_iterator = AsyncPageIterator(paged_func, page_size, page_token)
        self._data: Optional[List[T]] = None
        self._index = 0
        self._consecutive_empty_pages = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._data is None:
            self._data = await self._page_iterator.__anext__()
            self._record_page(self._data)

        while self._index >= len(self._data):
            if (
                self._consecutive_empty_pages >= MAX_CONSECUTIVE_EMPTY_PAGES
                and await self._page_iterator.get_next_page_token() is not None
            ):
                _raise_too_many_empty_pages()

            self._data = await self._page_iterator.__anext__()
            self._index = 0
            self._record_page(self._data)

        obj = self._data[self._index]
        self._index += 1
        return obj

    def _record_page(self, data: List[T]) -> None:
        self._consecutive_empty_pages = 0 if data else self._consecutive_empty_pages + 1
