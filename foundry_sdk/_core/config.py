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


from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

import httpx


def _default_timeout() -> httpx.Timeout:
    return httpx.Timeout(60.0, connect=10.0)


@dataclass
class Config:
    """
    Configuration for the HTTP session.
    """

    default_headers: Optional[Dict[str, str]] = None
    """HTTP headers to include with all requests."""

    proxies: Optional[Dict[Literal["http", "https"], str]] = None
    """Proxies to use for HTTP and HTTPS requests."""

    timeout: Optional[Union[int, float, httpx.Timeout]] = field(default_factory=_default_timeout)
    """
    The default timeout for all requests in seconds, or an ``httpx.Timeout`` for per-phase limits.
    Defaults to 10 seconds to connect and 60 seconds for each read, write, and connection pool
    wait. Set to ``None`` to disable all timeouts.
    """

    verify: Union[bool, str] = True
    """
    SSL verification, can be a boolean or a path to a CA bundle. When using an HTTPS proxy,
    connection this value will be passed to the proxy's SSL context as well.
    """

    default_params: Optional[Dict[str, Any]] = None
    """URL query parameters to include with all requests."""

    scheme: Literal["http", "https"] = "https"
    """URL scheme to use ('http' or 'https'). Defaults to 'https'."""

    max_retries: Optional[int] = None
    """
    The maximum number of times a failed request is retried.
    If no value is provided, it defaults to 4.
    """

    propagate_qos: Optional[Literal["AUTOMATIC_RETRY", "PROPAGATE_429_AND_503_TO_CALLER"]] = None
    """Indicates whether 429 and 503 status codes should be propagated as exceptions or retried."""

    backoff_slot_size_ms: Optional[int] = None
    """
    The size of one backoff time slot in milliseconds for call retries. For example,
    an exponential backoff retry algorithm may choose a backoff time in [0, backoffSlotSize * 2^c]
    for the c-th retry.
    If no value is provided, it defaults to 250 milliseconds.
    """
