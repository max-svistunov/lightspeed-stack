"""Networking helpers for building proxy- and TLS-aware HTTP clients.

Provides factory functions that construct httpx and aiohttp clients configured
with proxy and TLS settings from the application's NetworkingConfiguration.
These are used by client.py (llama-stack-client via httpx) and by aiohttp
consumers (Splunk, JWK, MCP OAuth probe).
"""

import fnmatch
import ssl
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import httpx
from llama_stack_client import DefaultAsyncHttpxClient

from log import get_logger
from models.config import NetworkingConfiguration
from utils.certificates import generate_ca_bundle
from utils.ssl_context import build_ssl_context
from utils.tls import TLSProfiles, TLSProtocolVersion

logger = get_logger(__name__)


def _parse_no_proxy(no_proxy: str) -> list[str]:
    """Parse a comma-separated no_proxy string into a list of patterns.

    Parameters:
        no_proxy: Comma-separated hostnames/IPs/patterns (e.g.,
            "localhost,127.0.0.1,.internal.corp").

    Returns:
        List of stripped, non-empty patterns.
    """
    return [p.strip() for p in no_proxy.split(",") if p.strip()]


def _host_matches_no_proxy(hostname: str, no_proxy_patterns: list[str]) -> bool:
    """Check if a hostname matches any no_proxy pattern.

    Supports exact matches, leading-dot domain matching (e.g., .corp.com
    matches sub.corp.com), and fnmatch glob patterns.

    Parameters:
        hostname: The hostname to check.
        no_proxy_patterns: List of no_proxy patterns.

    Returns:
        True if the hostname should bypass the proxy.
    """
    for pattern in no_proxy_patterns:
        if pattern == "*":
            return True
        if hostname == pattern:
            return True
        # Leading dot means "any subdomain of"
        if pattern.startswith(".") and hostname.endswith(pattern):
            return True
        if fnmatch.fnmatch(hostname, pattern):
            return True
    return False


def _resolve_ca_cert_path(networking: NetworkingConfiguration) -> Optional[Path]:
    """Resolve the CA certificate path, merging extra CAs if needed.

    If extra_ca paths are configured, generates a merged CA bundle.
    Otherwise returns the ca_cert_path from the TLS security profile.

    Parameters:
        networking: Networking configuration.

    Returns:
        Path to the CA certificate file to use, or None for system default.
    """
    if networking.extra_ca:
        cert_dir = networking.certificate_directory
        if cert_dir is None:
            cert_dir = Path("/tmp")
            logger.warning(
                "No certificate_directory configured; using /tmp for CA bundle. "
                "Set networking.certificate_directory for a persistent location."
            )
        bundle_path = generate_ca_bundle(networking.extra_ca, cert_dir)
        if bundle_path is not None:
            logger.info("Using merged CA bundle: %s", bundle_path)
            return bundle_path
        logger.warning("Failed to generate CA bundle, falling back to profile CA")

    # Fall back to TLS profile's ca_cert_path
    tls_profile = networking.tls_security_profile
    if tls_profile is not None and tls_profile.ca_cert_path is not None:
        return tls_profile.ca_cert_path
    return None


def _build_no_proxy_mounts(
    no_proxy: str,
) -> dict[str, None]:
    """Build httpx mounts dict for no_proxy bypass patterns.

    Parameters:
        no_proxy: Comma-separated no_proxy string.

    Returns:
        Dict mapping httpx mount patterns to None (bypass transport).
    """
    mounts: dict[str, None] = {}
    for pattern in _parse_no_proxy(no_proxy):
        if pattern == "*":
            mounts["all://"] = None
        elif pattern.startswith("."):
            mounts[f"all://*{pattern}"] = None
        else:
            mounts[f"all://{pattern}"] = None
    return mounts


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
    has_extra_ca = bool(networking.extra_ca)

    if not has_tls and not has_proxy and not has_skip_tls and not has_extra_ca:
        return None

    # Resolve CA certificate path (may generate merged bundle)
    ca_cert_path = _resolve_ca_cert_path(networking)

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
            ca_cert_path=ca_cert_path,
        )
    elif ca_cert_path is not None:
        # Extra CAs without a TLS profile — just use the bundle path
        verify = str(ca_cert_path)

    # Build proxy with no_proxy support
    proxy: Optional[str] = None
    if has_proxy and proxy_config is not None:
        proxy = proxy_config.https_proxy or proxy_config.http_proxy
        logger.info("Configuring httpx proxy: %s", proxy)

    logger.info("Creating custom httpx.AsyncClient with networking configuration")

    # Wire no_proxy via httpx mounts
    if proxy and proxy_config is not None and proxy_config.no_proxy:
        bypass_mounts = _build_no_proxy_mounts(proxy_config.no_proxy)
        all_mounts: dict[str, Optional[httpx.AsyncBaseTransport]] = {
            "all://": httpx.AsyncHTTPTransport(proxy=proxy, verify=verify),
            **bypass_mounts,
        }
        logger.info("Configured no_proxy bypasses: %s", list(bypass_mounts.keys()))
        return DefaultAsyncHttpxClient(verify=verify, mounts=all_mounts)

    return DefaultAsyncHttpxClient(verify=verify, proxy=proxy)


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

    # Resolve CA cert path
    ca_cert_path = _resolve_ca_cert_path(networking) if networking is not None else None

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
            ca_cert_path=ca_cert_path,
        )
        return aiohttp.TCPConnector(ssl=ssl_ctx)

    # extra_ca without a TLS profile — load the merged CA bundle into a
    # default SSLContext so aiohttp consumers trust the extra CAs
    if ca_cert_path is not None:
        ssl_ctx = ssl.create_default_context(cafile=str(ca_cert_path))
        return aiohttp.TCPConnector(ssl=ssl_ctx)

    return aiohttp.TCPConnector()


def get_aiohttp_proxy(
    networking: Optional[NetworkingConfiguration],
    target_url: Optional[str] = None,
) -> Optional[str]:
    """Extract the proxy URL for aiohttp, respecting no_proxy.

    Parameters:
        networking: Networking configuration, or None.
        target_url: The URL being requested. If it matches a no_proxy
            pattern, None is returned (bypass proxy).

    Returns:
        Proxy URL string, or None if no proxy is configured or the
        target matches no_proxy.
    """
    if networking is None or networking.proxy is None:
        return None

    proxy_config = networking.proxy
    proxy_url = proxy_config.https_proxy or proxy_config.http_proxy
    if proxy_url is None:
        return None

    # Check no_proxy
    if proxy_config.no_proxy and target_url:
        no_proxy_patterns = _parse_no_proxy(proxy_config.no_proxy)
        parsed = urlparse(target_url)
        hostname = parsed.hostname or ""
        if _host_matches_no_proxy(hostname, no_proxy_patterns):
            logger.debug("Bypassing proxy for %s (matches no_proxy)", hostname)
            return None

    return proxy_url
