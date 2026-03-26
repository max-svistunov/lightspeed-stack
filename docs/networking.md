# Networking Configuration

This guide covers how to configure proxy settings, TLS security profiles, and
custom CA certificates for outgoing connections from the Lightspeed Stack.

These settings apply to **all** outgoing HTTP/HTTPS connections:

- Llama Stack provider connections
- Splunk HEC telemetry
- JWK token endpoint fetching
- MCP server OAuth probing

## Quick Start

Add a `networking` section to your `lightspeed-stack.yaml`:

```yaml
networking:
  proxy:
    https_proxy: http://proxy.corp.example.com:8080
    no_proxy: localhost,127.0.0.1,.internal.corp
  tls_security_profile:
    type: IntermediateType
  extra_ca:
    - /etc/pki/tls/certs/corporate-ca.pem
```

## Proxy Configuration

Configure HTTP/HTTPS proxy for environments where direct internet access is
restricted (corporate firewalls, air-gapped networks).

```yaml
networking:
  proxy:
    http_proxy: http://proxy:8080     # Proxy for HTTP connections
    https_proxy: http://proxy:8080    # Proxy for HTTPS connections
    no_proxy: localhost,127.0.0.1     # Hosts that bypass the proxy
```

| Field | Type | Description |
|-------|------|-------------|
| `http_proxy` | string | Proxy URL for HTTP connections |
| `https_proxy` | string | Proxy URL for HTTPS connections (typically the same as http_proxy) |
| `no_proxy` | string | Comma-separated list of hostnames/IPs to bypass |

When the `proxy` section is not configured, the underlying HTTP libraries
(httpx, aiohttp) will respect the standard `HTTP_PROXY`, `HTTPS_PROXY`, and
`NO_PROXY` environment variables. When proxy settings are explicitly
configured, they take precedence over environment variables.

### Tunnel Proxy vs. Interception Proxy

**Tunnel proxy** (HTTP CONNECT): The proxy creates a TCP tunnel to the
destination. TLS is end-to-end between the client and server. No custom CA
certificate is needed.

```yaml
networking:
  proxy:
    https_proxy: http://tunnel-proxy:8080
```

**Interception proxy** (MITM/SSL inspection): The proxy terminates the TLS
connection, inspects traffic, and re-encrypts with its own certificate. You
must trust the proxy's CA certificate:

```yaml
networking:
  proxy:
    https_proxy: http://interception-proxy:8080
  tls_security_profile:
    type: IntermediateType
    caCertPath: /etc/pki/tls/certs/proxy-ca.pem
```

## TLS Security Profiles

TLS security profiles control the minimum TLS version and allowed cipher
suites for outgoing connections. Profiles are compatible with the
[OpenShift TLS security profile specification](https://docs.openshift.com/container-platform/latest/security/tls-security-profiles.html).

```yaml
networking:
  tls_security_profile:
    type: ModernType          # Profile type
    minTLSVersion: VersionTLS13   # Override minimum version (optional)
    caCertPath: /path/to/ca.pem   # Custom CA certificate (optional)
    skipTLSVerification: false    # Never enable in production
```

### Profile Types

| Profile | Min TLS Version | Ciphers | Use Case |
|---------|-----------------|---------|----------|
| `OldType` | TLS 1.0 | 29 ciphers (including legacy) | Legacy systems |
| `IntermediateType` | TLS 1.2 | 11 ciphers | **Recommended** for most deployments |
| `ModernType` | TLS 1.3 | 3 ciphers (TLS 1.3 only) | Maximum security |
| `Custom` | Configurable | User-defined | Special requirements |

### Custom Profile

For custom TLS settings, set `type: Custom` and provide your own cipher list
and minimum version:

```yaml
networking:
  tls_security_profile:
    type: Custom
    minTLSVersion: VersionTLS12
    ciphers:
      - ECDHE-RSA-AES128-GCM-SHA256
      - ECDHE-RSA-AES256-GCM-SHA384
```

### TLS Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | - | Profile type: `OldType`, `IntermediateType`, `ModernType`, `Custom` |
| `minTLSVersion` | string | Profile default | `VersionTLS10` through `VersionTLS13` |
| `ciphers` | list | Profile default | List of allowed cipher suite names |
| `caCertPath` | string | - | Path to CA certificate file |
| `skipTLSVerification` | boolean | `false` | Disable TLS verification (testing only) |

## Custom CA Certificates

For environments with internal CAs (e.g., interception proxies, self-signed
internal services), add extra CA certificates:

```yaml
networking:
  extra_ca:
    - /etc/pki/tls/certs/corporate-root-ca.pem
    - /etc/pki/tls/certs/intermediate-ca.pem
  certificate_directory: /var/lib/lightspeed/certs
```

The Lightspeed Stack merges these certificates with the system trust store
(certifi bundle) into a single CA bundle file. Duplicate certificates are
detected and skipped.

| Field | Type | Description |
|-------|------|-------------|
| `extra_ca` | list | Paths to PEM-encoded CA certificate files |
| `certificate_directory` | string | Directory for the merged CA bundle |

## Common Configurations

### Corporate proxy with internal CA

```yaml
networking:
  proxy:
    https_proxy: http://proxy.corp.example.com:8080
    no_proxy: localhost,127.0.0.1,.corp.example.com
  tls_security_profile:
    type: IntermediateType
    caCertPath: /etc/pki/tls/certs/corp-ca.pem
```

### TLS 1.3 only (maximum security)

```yaml
networking:
  tls_security_profile:
    type: ModernType
```

### No networking customization (default)

When the `networking` section is omitted entirely, the Lightspeed Stack uses
system defaults: no proxy (unless set via environment variables), system CA
trust store, and OpenSSL default TLS settings.

## Troubleshooting

**Connection refused through proxy**: Verify the proxy URL is correct and the
proxy server is running. Check that `no_proxy` does not accidentally include
the target host.

**Certificate verification failed**: Ensure the correct CA certificate is
specified in `caCertPath` or `extra_ca`. The certificate must be PEM-encoded.
Check that the certificate has not expired.

**TLS handshake failure**: The server may not support the minimum TLS version
or cipher suites required by the selected profile. Try a less restrictive
profile (e.g., `IntermediateType` instead of `ModernType`).
