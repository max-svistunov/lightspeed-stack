"""Integration tests for proxy and TLS networking.

These tests verify that the networking helpers correctly build HTTP clients
that route traffic through proxies and apply TLS profiles. They use real
async network connections with local test proxy servers.
"""

import asyncio
import threading
import time
from typing import Any

import httpx
import pytest

from models.config import (
    NetworkingConfiguration,
    ProxyConfiguration,
    TLSSecurityProfile,
)
from tests.e2e.proxy.tunnel_proxy import TunnelProxy
from utils.networking import build_httpx_client


@pytest.fixture(name="tunnel_proxy")
def tunnel_proxy_fixture() -> Any:
    """Start a tunnel proxy in a background thread and return it."""
    proxy = TunnelProxy(port=18888)
    loop = asyncio.new_event_loop()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proxy.start())
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1)

    yield proxy  # type: ignore[misc]

    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


class TestTunnelProxyIntegration:
    """Integration tests for tunnel proxy routing."""

    @pytest.mark.asyncio
    async def test_httpx_client_routes_through_tunnel_proxy(
        self, tunnel_proxy: TunnelProxy
    ) -> None:
        """Test that build_httpx_client creates a client that routes through proxy."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                https_proxy=f"http://{tunnel_proxy.host}:{tunnel_proxy.port}"
            )
        )
        client = build_httpx_client(nc)
        assert client is not None

        # Make a request to an HTTPS endpoint through the proxy
        try:
            await client.get("https://httpbin.org/get", timeout=10)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # Connection may fail (httpbin may be unreachable), but the
            # proxy should still have seen the CONNECT request
            pass

        assert (
            tunnel_proxy.connect_count >= 1
        ), f"Expected proxy to handle CONNECT, got {tunnel_proxy.connect_count}"
        assert tunnel_proxy.last_connect_target is not None
        assert "httpbin.org" in tunnel_proxy.last_connect_target

    def test_no_proxy_returns_none(self) -> None:
        """Test that no proxy config returns None (no customization)."""
        nc = NetworkingConfiguration()
        assert build_httpx_client(nc) is None


class TestTLSProfileIntegration:
    """Integration tests for TLS profile application."""

    def test_httpx_client_with_modern_profile(self) -> None:
        """Test that ModernType profile creates a working client."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(
                profile_type="ModernType",
                min_tls_version="VersionTLS13",
            )
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_httpx_client_with_skip_verification(self) -> None:
        """Test that skip_tls_verification creates a client."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(skip_tls_verification=True)
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_no_networking_config_returns_none(self) -> None:
        """Test that no networking config returns None (default behavior)."""
        assert build_httpx_client(None) is None
        assert build_httpx_client(NetworkingConfiguration()) is None
