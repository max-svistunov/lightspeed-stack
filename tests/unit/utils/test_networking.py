"""Unit tests for networking helpers defined in src/utils/networking.py."""

import ssl

import pytest
from pytest_mock import MockerFixture

from models.config import (
    NetworkingConfiguration,
    ProxyConfiguration,
    TLSSecurityProfile,
)
from utils.networking import (
    _host_matches_no_proxy,
    _parse_no_proxy,
    build_aiohttp_connector,
    build_httpx_client,
    get_aiohttp_proxy,
)


class TestBuildHttpxClient:
    """Tests for build_httpx_client function."""

    def test_returns_none_when_no_config(self) -> None:
        """Test that None config returns None."""
        assert build_httpx_client(None) is None

    def test_returns_none_when_empty_config(self) -> None:
        """Test that empty networking config returns None."""
        assert build_httpx_client(NetworkingConfiguration()) is None

    def test_returns_client_with_proxy(self, mocker: MockerFixture) -> None:
        """Test that proxy config creates a client."""
        mocker.patch("utils.networking.build_ssl_context")
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(https_proxy="http://proxy:8080")
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_returns_client_with_tls_profile(self, mocker: MockerFixture) -> None:
        """Test that TLS profile creates a client."""
        mock_ctx = mocker.MagicMock(spec=ssl.SSLContext)
        mocker.patch("utils.networking.build_ssl_context", return_value=mock_ctx)
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(profile_type="ModernType")
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_returns_client_with_skip_verification(self) -> None:
        """Test that skip_tls_verification creates a client."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(skip_tls_verification=True)
        )
        client = build_httpx_client(nc)
        assert client is not None

    def test_tls_profile_none_type_returns_none(self) -> None:
        """Test that TLS profile with None type is treated as no customization."""
        nc = NetworkingConfiguration(tls_security_profile=TLSSecurityProfile())
        assert build_httpx_client(nc) is None


class TestBuildAiohttpConnector:
    """Tests for build_aiohttp_connector function."""

    @pytest.mark.asyncio
    async def test_default_connector_when_no_config(self) -> None:
        """Test that None config creates a default connector."""
        connector = build_aiohttp_connector(None)
        assert connector is not None

    @pytest.mark.asyncio
    async def test_connector_with_skip_verification(self) -> None:
        """Test that skip_tls_verification disables SSL."""
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(skip_tls_verification=True)
        )
        connector = build_aiohttp_connector(nc)
        assert connector is not None

    @pytest.mark.asyncio
    async def test_connector_with_tls_profile(self, mocker: MockerFixture) -> None:
        """Test that TLS profile configures SSL context."""
        mock_ctx = mocker.MagicMock(spec=ssl.SSLContext)
        mocker.patch("utils.networking.build_ssl_context", return_value=mock_ctx)
        nc = NetworkingConfiguration(
            tls_security_profile=TLSSecurityProfile(profile_type="ModernType")
        )
        connector = build_aiohttp_connector(nc)
        assert connector is not None


class TestGetAiohttpProxy:
    """Tests for get_aiohttp_proxy function."""

    def test_returns_none_when_no_config(self) -> None:
        """Test that None config returns None."""
        assert get_aiohttp_proxy(None) is None

    def test_returns_none_when_no_proxy(self) -> None:
        """Test that config without proxy returns None."""
        assert get_aiohttp_proxy(NetworkingConfiguration()) is None

    def test_returns_https_proxy(self) -> None:
        """Test that https_proxy is preferred."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                http_proxy="http://http-proxy:8080",
                https_proxy="http://https-proxy:8080",
            )
        )
        assert get_aiohttp_proxy(nc) == "http://https-proxy:8080"

    def test_falls_back_to_http_proxy(self) -> None:
        """Test that http_proxy is used when https_proxy is not set."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(http_proxy="http://http-proxy:8080")
        )
        assert get_aiohttp_proxy(nc) == "http://http-proxy:8080"

    def test_no_proxy_bypasses_matching_host(self) -> None:
        """Test that no_proxy bypasses matching target URLs."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                https_proxy="http://proxy:8080",
                no_proxy="localhost,127.0.0.1",
            )
        )
        assert get_aiohttp_proxy(nc, target_url="http://localhost:8321") is None
        assert get_aiohttp_proxy(nc, target_url="http://127.0.0.1:8321") is None

    def test_no_proxy_allows_non_matching_host(self) -> None:
        """Test that non-matching hosts still use the proxy."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                https_proxy="http://proxy:8080",
                no_proxy="localhost",
            )
        )
        assert (
            get_aiohttp_proxy(nc, target_url="https://api.openai.com")
            == "http://proxy:8080"
        )

    def test_no_proxy_domain_suffix(self) -> None:
        """Test that .domain patterns match subdomains."""
        nc = NetworkingConfiguration(
            proxy=ProxyConfiguration(
                https_proxy="http://proxy:8080",
                no_proxy=".internal.corp",
            )
        )
        assert get_aiohttp_proxy(nc, target_url="https://api.internal.corp") is None
        assert (
            get_aiohttp_proxy(nc, target_url="https://external.com")
            == "http://proxy:8080"
        )


class TestParseNoProxy:
    """Tests for _parse_no_proxy function."""

    def test_splits_comma_separated(self) -> None:
        """Test basic comma splitting."""
        assert _parse_no_proxy("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self) -> None:
        """Test whitespace is stripped."""
        assert _parse_no_proxy(" a , b , c ") == ["a", "b", "c"]

    def test_empty_string(self) -> None:
        """Test empty string returns empty list."""
        assert _parse_no_proxy("") == []


class TestHostMatchesNoProxy:
    """Tests for _host_matches_no_proxy function."""

    def test_exact_match(self) -> None:
        """Test exact hostname match."""
        assert _host_matches_no_proxy("localhost", ["localhost"]) is True

    def test_no_match(self) -> None:
        """Test non-matching hostname."""
        assert _host_matches_no_proxy("external.com", ["localhost"]) is False

    def test_wildcard_matches_all(self) -> None:
        """Test that * matches everything."""
        assert _host_matches_no_proxy("anything.com", ["*"]) is True

    def test_dot_prefix_matches_subdomains(self) -> None:
        """Test that .domain matches subdomains."""
        assert _host_matches_no_proxy("sub.corp.com", [".corp.com"]) is True
        assert _host_matches_no_proxy("corp.com", [".corp.com"]) is False
