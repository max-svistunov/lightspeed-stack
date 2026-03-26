"""Step definitions for proxy and TLS networking e2e tests."""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import trustme
import yaml
from behave import given, then, when  # pyright: ignore[reportAttributeAccessIssue]
from behave.runner import Context


def _get_default_config_path(context: Context) -> str:
    """Get the path to the default lightspeed-stack configuration."""
    mode_dir = "library-mode" if context.is_library_mode else "server-mode"
    return f"tests/e2e/configuration/{mode_dir}/lightspeed-stack.yaml"


def _load_config(config_path: str) -> dict:
    """Load a YAML config file, overriding hostnames for local testing."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override Llama Stack URL with environment variable for local testing
    llama_host = os.getenv("E2E_LLAMA_HOSTNAME", "localhost")
    llama_port = os.getenv("E2E_LLAMA_PORT", "8321")
    llama_url = os.getenv("E2E_LLAMA_STACK_URL", f"http://{llama_host}:{llama_port}")
    if "llama_stack" in config:
        config["llama_stack"]["url"] = llama_url

    # Strip MCP servers for proxy tests (they use Docker hostnames)
    config.pop("mcp_servers", None)

    return config


def _write_config(config: dict, path: str) -> None:
    """Write a YAML config file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def _restart_lightspeed_stack(config_path: str) -> None:
    """Restart the lightspeed stack with a new config.

    Kills the existing process, writes the new config, and starts fresh.
    """
    # Kill existing lightspeed stack
    subprocess.run(
        ["pkill", "-f", "lightspeed_stack.py"],
        capture_output=True,
        check=False,
    )
    time.sleep(2)

    # Start with new config
    env = os.environ.copy()
    env["OPENSSL_CONF"] = ""  # Workaround for OpenSSL 3.5.x init issues
    log_path = "/tmp/lightspeed-stack-proxy-test.log"
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            ["uv", "run", "src/lightspeed_stack.py", "-c", config_path],
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

    # Wait for readiness
    for i in range(30):
        try:
            hostname = os.getenv("E2E_LSC_HOSTNAME", "localhost")
            port = os.getenv("E2E_LSC_PORT", "8080")
            resp = requests.get(f"http://{hostname}:{port}/liveness", timeout=2)
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)

    raise TimeoutError("Lightspeed stack did not start within 60 seconds")


# --- Tunnel Proxy Steps ---


@given("A tunnel proxy is running on port {port:d}")
def start_tunnel_proxy(context: Context, port: int) -> None:
    """Start a tunnel proxy in a background thread.

    Parameters:
        context: Behave context.
        port: Port number for the tunnel proxy.
    """
    from tests.e2e.proxy.tunnel_proxy import TunnelProxy

    proxy = TunnelProxy(port=port)

    loop = asyncio.new_event_loop()
    context.proxy_loop = loop
    context.tunnel_proxy = proxy

    import threading

    def run_proxy() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proxy.start())
        loop.run_forever()

    thread = threading.Thread(target=run_proxy, daemon=True)
    thread.start()
    time.sleep(1)  # Give proxy time to bind


@given("The lightspeed-stack is configured to use the tunnel proxy")
def configure_tunnel_proxy(context: Context) -> None:
    """Configure lightspeed-stack with tunnel proxy settings.

    Parameters:
        context: Behave context with tunnel_proxy attribute.
    """
    proxy = context.tunnel_proxy
    config_path = _get_default_config_path(context)
    config = _load_config(config_path)

    config["networking"] = {
        "proxy": {
            "https_proxy": f"http://{proxy.host}:{proxy.port}",
            "no_proxy": "localhost,127.0.0.1",
        }
    }

    proxy_config_path = os.path.join(tempfile.gettempdir(), "lsc-proxy-config.yaml")
    _write_config(config, proxy_config_path)
    context.proxy_config_path = proxy_config_path

    _restart_lightspeed_stack(proxy_config_path)


@then("The tunnel proxy handled at least {count:d} CONNECT request")
def verify_tunnel_proxy_used(context: Context, count: int) -> None:
    """Verify the tunnel proxy received CONNECT requests.

    Parameters:
        context: Behave context with tunnel_proxy attribute.
        count: Minimum expected CONNECT request count.
    """
    proxy = context.tunnel_proxy
    assert (
        proxy.connect_count >= count
    ), f"Expected at least {count} CONNECT requests, got {proxy.connect_count}"


# --- Interception Proxy Steps ---


@given("An interception proxy with trustme CA is running on port {port:d}")
def start_interception_proxy(context: Context, port: int) -> None:
    """Start an interception proxy with trustme CA.

    Parameters:
        context: Behave context.
        port: Port number for the interception proxy.
    """
    from tests.e2e.proxy.interception_proxy import InterceptionProxy

    ca = trustme.CA()
    proxy = InterceptionProxy(ca=ca, port=port)

    # Export CA cert for the lightspeed-stack to trust
    ca_cert_path = Path(tempfile.gettempdir()) / "interception-proxy-ca.pem"
    proxy.export_ca_cert(ca_cert_path)

    loop = asyncio.new_event_loop()
    context.interception_proxy_loop = loop
    context.interception_proxy = proxy
    context.interception_ca_cert_path = str(ca_cert_path)

    import threading

    def run_proxy() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proxy.start())
        loop.run_forever()

    thread = threading.Thread(target=run_proxy, daemon=True)
    thread.start()
    time.sleep(1)


@given("The lightspeed-stack is configured to use the interception proxy with CA cert")
def configure_interception_proxy(context: Context) -> None:
    """Configure lightspeed-stack with interception proxy and CA cert.

    Parameters:
        context: Behave context with interception_proxy attribute.
    """
    proxy = context.interception_proxy
    config_path = _get_default_config_path(context)
    config = _load_config(config_path)

    config["networking"] = {
        "proxy": {
            "https_proxy": f"http://{proxy.host}:{proxy.port}",
        },
        "tls_security_profile": {
            "type": "IntermediateType",
            "caCertPath": context.interception_ca_cert_path,
        },
    }

    proxy_config_path = os.path.join(
        tempfile.gettempdir(), "lsc-interception-config.yaml"
    )
    _write_config(config, proxy_config_path)
    context.proxy_config_path = proxy_config_path

    _restart_lightspeed_stack(proxy_config_path)


@then("The interception proxy intercepted at least {count:d} connection")
def verify_interception_proxy_used(context: Context, count: int) -> None:
    """Verify the interception proxy intercepted connections.

    Parameters:
        context: Behave context with interception_proxy attribute.
        count: Minimum expected intercepted connection count.
    """
    proxy = context.interception_proxy
    assert (
        proxy.connect_count >= count
    ), f"Expected at least {count} intercepted connections, got {proxy.connect_count}"


# --- TLS Profile Steps ---


@given('The lightspeed-stack is configured with TLS profile "{profile_type}"')
def configure_tls_profile(context: Context, profile_type: str) -> None:
    """Configure lightspeed-stack with a TLS security profile.

    Parameters:
        context: Behave context.
        profile_type: TLS profile type name.
    """
    config_path = _get_default_config_path(context)
    config = _load_config(config_path)

    config["networking"] = {
        "tls_security_profile": {
            "type": profile_type,
        }
    }

    tls_config_path = os.path.join(tempfile.gettempdir(), "lsc-tls-config.yaml")
    _write_config(config, tls_config_path)
    context.proxy_config_path = tls_config_path

    _restart_lightspeed_stack(tls_config_path)


# --- Negative Test Steps ---


@given('The lightspeed-stack is configured with unreachable proxy "{proxy_url}"')
def configure_unreachable_proxy(context: Context, proxy_url: str) -> None:
    """Configure lightspeed-stack with a proxy that cannot be reached.

    Parameters:
        context: Behave context.
        proxy_url: URL of the unreachable proxy.
    """
    config_path = _get_default_config_path(context)
    config = _load_config(config_path)

    config["networking"] = {
        "proxy": {
            "https_proxy": proxy_url,
        }
    }

    neg_config_path = os.path.join(
        tempfile.gettempdir(), "lsc-unreachable-proxy-config.yaml"
    )
    _write_config(config, neg_config_path)
    context.proxy_config_path = neg_config_path

    _restart_lightspeed_stack(neg_config_path)


@when('I send a query "{query}" and expect failure')
def send_query_expect_failure(context: Context, query: str) -> None:
    """Send a query and capture the response, expecting failure.

    Parameters:
        context: Behave context.
        query: Query string to send.
    """
    hostname = context.hostname
    port = context.port
    try:
        context.response = requests.post(
            f"http://{hostname}:{port}/v1/query",
            json={"query": query},
            timeout=30,
        )
    except requests.ConnectionError as e:
        context.connection_error = str(e)
        context.response = None


@then("The response indicates a connection error")
def verify_connection_error(context: Context) -> None:
    """Verify that the response indicates a connection error.

    Parameters:
        context: Behave context.
    """
    if context.response is not None:
        # If we got a response, it should be a 5xx error
        assert (
            context.response.status_code >= 500
        ), f"Expected 5xx error, got {context.response.status_code}"
    else:
        # Connection error is also acceptable
        assert hasattr(
            context, "connection_error"
        ), "Expected a connection error or 5xx response"
