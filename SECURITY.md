# Security Policy

## Supported Versions

OpenWarden is currently an alpha project. Security fixes are applied to the
latest published version.

## Reporting a Vulnerability

Use GitHub's private vulnerability reporting for this repository. If that is
unavailable, email Ahmad Hammad at Ahmad.Hammad@ieee.org.

Do not open a public issue containing credentials, private prompts, customer
data, model output, or exploit details. Reports should include the affected
version, impact, reproduction steps, and any suggested mitigation.

## Security Boundaries

OpenWarden is an application-layer guardrail and is not a replacement for:

- Deterministic authorization and access control.
- Provider-side safety systems.
- Secret storage and transport security.
- Sandboxing for tool execution.
- Document-level authorization in retrieval systems.

Keep retrieval authorization in the retriever or data layer. Treat retrieved
documents as untrusted content. Use `fail_open=False` for enforcement-sensitive
workflows.

OpenWarden does not log raw prompts or responses by default. Applications that
add content logging are responsible for protecting that data.
