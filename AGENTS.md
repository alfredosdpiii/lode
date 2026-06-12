# Lode Agent Guide

## Project

Lode is a local-first Python repository knowledge graph. It ships the `lode` CLI
and `loded` local HTTP daemon from `src/lode`.

## Setup

```bash
uv sync --extra kuzu --dev
```

Run the daemon stack:

```bash
docker compose up -d --build
```

## Validation

Run these before committing code changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests scripts benchmarks
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv run radon cc src tests scripts -s -n B
uv run vulture src tests scripts --min-confidence 90
uv run deptry .
uv run bandit -c pyproject.toml -r src scripts -q
uv run python scripts/check_large_files.py
uv run python scripts/check_todos.py
uv run python scripts/check_agents_md.py
uv run python scripts/check_feature_flags.py
uv run python scripts/check_duplicate_code.py
uv run python scripts/generate_openapi_docs.py --check
```

Use `RUN_LODE_DOCKER_E2E=1 uv run python -m unittest tests.test_e2e -v` only
when Docker is available and a full Compose smoke test is needed.

## Conventions

- Keep Lode local-first. Do not add hosted telemetry or external services by default.
- Prefer JSON output for agent-facing commands and include path/line citations.
- Use `snake_case` for functions, variables, and modules. Use `PascalCase` for classes.
- Keep public CLI and daemon behavior backward compatible unless the change is explicit.
- New maintenance comments must link an issue, for example `TODO(#123)`.

## Security

- Never commit secrets. Keep `.env` files local and use `.env.example` for templates.
- Redact tokens, passwords, and keys before logging.
- Treat repository source as sensitive. Lode should not upload indexed content by default.

## Release

PyPI publishing is handled by `.github/workflows/publish.yml` for `v*` tags or
manual dispatch. Update `pyproject.toml`, `src/lode/__init__.py`, `uv.lock`, and
`tests/smoke_test.py` together when bumping versions.
