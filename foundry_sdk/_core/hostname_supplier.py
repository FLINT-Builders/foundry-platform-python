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


import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

log = logging.getLogger(__name__)


class EndpointType(Enum):
    GENERIC = "GENERIC"
    HIGH_SCALE = "HIGH_SCALE"
    AUTH = "AUTH"


class HostnameSupplier(ABC):
    @abstractmethod
    def get_hostname(self) -> str:
        """Return the base hostname (e.g., 'https://example.com')."""
        ...

    @abstractmethod
    def get_endpoint(self, endpoint_type: EndpointType) -> str:
        """Return a base URL including scheme for an endpoint-specific URL."""
        ...

    @property
    @abstractmethod
    def is_user_supplied(self) -> bool:
        """Return True if the user explicitly provided their own hostname."""
        ...


class StaticHostnameSupplier(HostnameSupplier):
    def __init__(self, base_url: str, is_user_supplied: bool) -> None:
        self._base_url = base_url
        self._api_gateway_url = base_url + "/api"
        self._multipass_url = base_url + "/multipass/api"
        self._stream_proxy_url = base_url + "/api"
        self._is_user_supplied = is_user_supplied

    def get_hostname(self) -> str:
        return self._base_url

    def get_endpoint(self, endpoint_type: EndpointType) -> str:
        if endpoint_type == EndpointType.GENERIC:
            return self._api_gateway_url
        elif endpoint_type == EndpointType.AUTH:
            return self._multipass_url
        elif endpoint_type == EndpointType.HIGH_SCALE:
            return self._stream_proxy_url

        raise ValueError(f"Unsupported endpoint type: {endpoint_type}")

    @property
    def is_user_supplied(self) -> bool:
        return self._is_user_supplied


_DEFAULT_PORTS = {"http": 80, "https": 443}

_Origin = Tuple[str, str, int]


def _parse_origin(url: str) -> Optional[_Origin]:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    return parts.scheme, parts.hostname, port if port is not None else _DEFAULT_PORTS[parts.scheme]


def _format_origin(origin: _Origin) -> str:
    scheme, host, port = origin
    if ":" in host:
        host = f"[{host}]"
    if port != _DEFAULT_PORTS[scheme]:
        host = f"{host}:{port}"
    return f"{scheme}://{host}"


class ServiceDiscoveryHostnameSupplier(HostnameSupplier):
    """Resolves endpoints from a service discovery mapping.

    Discovered URLs are only used when they share the origin (scheme, host and port) of the
    fallback supplier's hostname, or, without a fallback, use the configured scheme. URLs with
    userinfo, a query or a fragment are ignored. Endpoints are resolved once, at construction.
    """

    _SERVICE_NAMES = {
        EndpointType.GENERIC: "api-gateway",
        EndpointType.AUTH: "multipass",
        EndpointType.HIGH_SCALE: "stream-proxy",
    }

    def __init__(
        self,
        services: Any,
        fallback_supplier: HostnameSupplier | None = None,
        scheme: str = "https",
    ) -> None:
        self._fallback_supplier = fallback_supplier
        self._scheme = scheme
        self._allowed_origin = (
            _parse_origin(fallback_supplier.get_hostname())
            if fallback_supplier is not None
            else None
        )
        if not isinstance(services, dict):
            services = {}
        self._endpoints: Dict[EndpointType, Optional[str]] = {
            endpoint_type: self._find_service_url(services, service_name)
            for endpoint_type, service_name in self._SERVICE_NAMES.items()
        }

    def get_hostname(self) -> str:
        if self._fallback_supplier is not None:
            return self._fallback_supplier.get_hostname()
        raise ValueError(
            "Unable to provide hostname. Please provide a hostname parameter to the client."
        )

    def get_endpoint(self, endpoint_type: EndpointType) -> str:
        service_name = self._SERVICE_NAMES.get(endpoint_type)
        if service_name is None:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")

        url = self._endpoints[endpoint_type]
        if url is not None:
            return url

        if self._fallback_supplier is not None:
            return self._fallback_supplier.get_endpoint(endpoint_type)
        raise ValueError(f"Unable to discover service '{service_name}'.")

    @property
    def is_user_supplied(self) -> bool:
        return False

    def _find_service_url(self, services: Dict[Any, Any], service_name: str) -> Optional[str]:
        # Functions swaps hyphens for underscores in service names
        alt_service_name = service_name.replace("-", "_")
        service_names = [service_name, alt_service_name]
        for name in service_names:
            urls = services.get(name)
            if not isinstance(urls, list):
                continue
            for url in urls:
                validated_url = self._validate_service_url(url)
                if validated_url is not None:
                    return validated_url
                log.warning(
                    "Ignoring service discovery URL for '%s': it must use the origin of the "
                    "configured Foundry hostname and must not contain userinfo, a query or a "
                    "fragment.",
                    name,
                )

        return None

    def _validate_service_url(self, url: Any) -> Optional[str]:
        if not isinstance(url, str):
            return None
        try:
            parts = urlsplit(url)
        except ValueError:
            return None
        if "@" in parts.netloc or parts.query or parts.fragment:
            return None

        origin = _parse_origin(url)
        if origin is None:
            return None
        if self._fallback_supplier is not None:
            if self._allowed_origin is None or origin != self._allowed_origin:
                return None
        elif origin[0] != self._scheme:
            return None

        return _format_origin(origin) + parts.path.rstrip("/")
