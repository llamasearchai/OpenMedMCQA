# Contributing Guide

Thanks for your interest in improving MedAgent!

## Development Setup

- Python 3.11 recommended
- Install dependencies: `uv pip install -e ".[dev,cli,data]"`
- Install pre-commit hooks: `pre-commit install`

## Commands

- Lint: `ruff check src tests` / Format check: `ruff format --check src tests`
- Types: `mypy src`
- Tests: `pytest -q --disable-warnings --maxfail=1`
- Coverage: `make test-cov`
- All via tox: `make test-all`

## Pull Request Checklist

- [ ] Add/Update tests for new behavior
- [ ] Run linters and type checks
- [ ] Update docs/README/TESTING as needed
- [ ] Note security/privacy considerations

## Architecture Notes

- Core modules live under `src/medagent/`.
- Keep functions small and side-effect free when possible.
- Prefer explicit types; mypy runs in strict mode.

## Security & Privacy

- Do not include secrets in code or logs.
- Avoid adding endpoints that bypass auth; consult SECURITY.md.

## Code of Conduct

- Be respectful, constructive, and inclusive. Report issues privately for sensitive matters.

