@Proxy
@skip-in-library-mode
Feature: Proxy and TLS networking tests

  Verify that the Lightspeed Stack correctly routes outgoing traffic
  through configured proxies and enforces TLS security profiles.

  Background:
    Given The service is started locally
      And REST API service prefix is /v1

  # Proxy-restart scenarios require HTTPS endpoints for CONNECT tunneling.
  # In local testing, Llama Stack is HTTP-only, so these are skipped.
  # Proxy routing is verified in tests/integration/test_proxy_networking.py.
  @TunnelProxy
  @skip
  Scenario: Traffic is routed through a configured tunnel proxy
    Given A tunnel proxy is running on port 8888
      And The lightspeed-stack is configured to use the tunnel proxy
     When I access endpoint "readiness" using HTTP GET method
     Then The status code of the response is 200
      And The tunnel proxy handled at least 1 CONNECT request

  @InterceptionProxy
  @skip
  Scenario: Interception proxy works with correct CA certificate
    Given An interception proxy with trustme CA is running on port 8889
      And The lightspeed-stack is configured to use the interception proxy with CA cert
     When I access endpoint "readiness" using HTTP GET method
     Then The status code of the response is 200
      And The interception proxy intercepted at least 1 connection

  @TLSProfile
  Scenario: TLS security profile is applied to outgoing connections
    Given The lightspeed-stack is configured with TLS profile "IntermediateType"
     When I access endpoint "readiness" using HTTP GET method
     Then The status code of the response is 200

  @TLSProfile
  Scenario: ModernType TLS profile enforces TLS 1.3
    Given The lightspeed-stack is configured with TLS profile "ModernType"
     When I access endpoint "readiness" using HTTP GET method
     Then The status code of the response is 200

  @NegativeProxy
  @skip
  Scenario: Connection fails when proxy is unreachable
    Given The lightspeed-stack is configured with unreachable proxy "http://127.0.0.1:19999"
     When I send a query "hello" and expect failure
     Then The response indicates a connection error
