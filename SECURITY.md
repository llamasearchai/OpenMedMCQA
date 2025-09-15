# Security Policy

## Supported Versions

This project uses semantic versioning. We aim to support the latest minor release. Security fixes are backported when practical.

## Reporting a Vulnerability

- Email the maintainers or open a private advisory on GitHub (Security > Advisories) with a clear description, impact, and PoC if available.
- Please do not open a public issue for sensitive reports.

## Hardening Guidelines

- Configure `MEDAGENT_API_KEY` in production to protect `/info`, `/ingest`, `/index`, `/ask`.
- Restrict `MEDAGENT_DATA_DIR` to a read-only, curated location; avoid sharing host paths unless necessary.
- Set `MEDAGENT_MAX_INGEST_BYTES` to a conservative limit based on operational needs.
- Limit CORS with `MEDAGENT_CORS_ORIGINS` to explicit origins; avoid `*` in production.
- Run behind a reverse proxy with TLS termination and optional WAF/DoS protections.
- Consider enabling rate limiting and request size limits at the proxy layer.
- Do not ingest PHI or PII; this project is not HIPAA-compliant by default.

## Secrets Management

- Provide `OPENAI_API_KEY` via environment variables or a secret manager.
- Never expose secrets via `/info`; the app redacts `openai_api_key` when set.
- Rotate keys regularly and restrict key permissions where possible.

## Dependency Hygiene

- Use CI to run lint/type/tests.
- Periodically audit dependencies and update minor/patch releases.
- For Docker, prefer multi-stage builds and image scanning.

