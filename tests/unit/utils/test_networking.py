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
