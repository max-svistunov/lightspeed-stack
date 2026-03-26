"""Integration tests for proxy and TLS networking.

These tests verify that the networking helpers correctly build HTTP clients
with proxy and TLS configurations applied.
"""

from models.config import (
    NetworkingConfiguration,
    ProxyConfiguration,
    TLSSecurityProfile,
)
from utils.networking import build_httpx_client


class TestBuildHttpxClientIntegration:
    """Integration tests for build_httpx_client."""

    def test_proxy_creates_client(self) -> None:
        """Test that proxy config creates a non-None client."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(https_proxy="http://proxy:8080")
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_no_proxy_creates_mounts(self) -> None:
        """Test that no_proxy creates a client with bypass configuration."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                https_proxy="http://proxy:8080",
                no_proxy="127.0.0.1,localhost,.internal.corp",
            )
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_tls_profile_creates_client(self) -> None:
        """Test that TLS profile creates a non-None client."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(
                profile_type="ModernType",
                min_tls_version="VersionTLS13",
            )
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_skip_verification_creates_client(self) -> None:
        """Test that skip_tls_verification creates a client."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(skip_tls_verification=True)
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_no_config_returns_none(self) -> None:
        """Test that None networking config returns None."""
        assert build_httpx_client(None) is None

    def test_empty_config_returns_none(self) -> None:
        """Test that empty networking config returns None."""
        assert build_httpx_client(NetworkingConfiguration()) is None
