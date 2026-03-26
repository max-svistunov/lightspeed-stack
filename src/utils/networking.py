"""Networking helpers for building proxy- and TLS-aware HTTP clients.

Provides factory functions that construct httpx and aiohttp clients configured
with proxy and TLS settings from the application's NetworkingConfiguration.
These are used by client.py (llama-stack-client via httpx) and by aiohttp
consumers (Splunk, JWK, MCP OAuth probe).
"""

import ssl
from typing import Optional

import aiohttp
import httpx
from llama_stack_client import DefaultAsyncHttpxClient

from log import get_logger
from models.config import NetworkingConfiguration
from utils.ssl_context import build_ssl_context
from utils.tls import TLSProfiles, TLSProtocolVersion

logger = get_logger(__name__)


def build_httpx_client(
    networking: Optional[NetworkingConfiguration],
) -> Optional[httpx.AsyncClient]:
    """Build an httpx AsyncClient with proxy and TLS from networking config.

    Uses DefaultAsyncHttpxClient from llama-stack-client to preserve SDK
    defaults (timeouts, connection limits, follow_redirects).

    Parameters:
        networking: Networking configuration, or None for no customization.

    Returns:
        A configured httpx.AsyncClient, or None if no networking config
        is provided (caller should let the SDK use its own defaults).
    """
    if networking is None:
        return None

    tls_profile = networking.tls_security_profile
    proxy_config = networking.proxy

    # Determine if any customization is needed
    has_tls = tls_profile is not None and tls_profile.profile_type is not None
    has_proxy = proxy_config is not None and (
        proxy_config.http_proxy or proxy_config.https_proxy
    )
    has_skip_tls = tls_profile is not None and tls_profile.skip_tls_verification

    if not has_tls and not has_proxy and not has_skip_tls:
        return None

    # Build SSL context or verification setting
    verify: ssl.SSLContext | bool | str = True
    if has_skip_tls:
        logger.warning(
            "TLS verification disabled for outgoing connections. "
            "This is insecure and should only be used for testing."
        )
        verify = False
    elif has_tls and tls_profile is not None:
        profile_type = TLSProfiles(tls_profile.profile_type)
        min_ver = (
            TLSProtocolVersion(tls_profile.min_tls_version)
            if tls_profile.min_tls_version
            else None
        )
        verify = build_ssl_context(
            profile_type=profile_type,
            min_tls_version=min_ver,
            ciphers=tls_profile.ciphers,
            ca_cert_path=tls_profile.ca_cert_path,
        )

    # Build proxy URL
    proxy: Optional[str] = None
    if has_proxy and proxy_config is not None:
        proxy = proxy_config.https_proxy or proxy_config.http_proxy
        logger.info("Configuring httpx proxy: %s", proxy)

    logger.info("Creating custom httpx.AsyncClient with networking configuration")
    return DefaultAsyncHttpxClient(
        verify=verify,
        proxy=proxy,
    )


def build_aiohttp_connector(
    networking: Optional[NetworkingConfiguration],
) -> aiohttp.TCPConnector:
    """Build an aiohttp TCPConnector with TLS settings from networking config.

    Parameters:
        networking: Networking configuration, or None for defaults.

    Returns:
        A configured aiohttp.TCPConnector.
    """
    tls_profile = networking.tls_security_profile if networking is not None else None

    if tls_profile is not None and tls_profile.skip_tls_verification:
        logger.warning(
            "TLS verification disabled for aiohttp connections. "
            "This is insecure and should only be used for testing."
        )
        return aiohttp.TCPConnector(ssl=False)

    if tls_profile is not None and tls_profile.profile_type is not None:
        profile_type = TLSProfiles(tls_profile.profile_type)
        min_ver = (
            TLSProtocolVersion(tls_profile.min_tls_version)
            if tls_profile.min_tls_version
            else None
        )
        ssl_ctx = build_ssl_context(
            profile_type=profile_type,
            min_tls_version=min_ver,
            ciphers=tls_profile.ciphers,
            ca_cert_path=tls_profile.ca_cert_path,
        )
        return aiohttp.TCPConnector(ssl=ssl_ctx)

    return aiohttp.TCPConnector()


def get_aiohttp_proxy(
    networking: Optional[NetworkingConfiguration],
) -> Optional[str]:
    """Extract the proxy URL for aiohttp from networking config.

    Parameters:
        networking: Networking configuration, or None.

    Returns:
        Proxy URL string, or None if no proxy is configured.
    """
    if networking is None or networking.proxy is None:
        return None
    return networking.proxy.https_proxy or networking.proxy.http_proxy
