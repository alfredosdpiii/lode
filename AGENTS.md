# Lode Agent Guide

## Project

Lode is a local-first Rust repository knowledge graph. It ships the `lode` CLI
and `loded` local HTTP daemon from `src/`.

For Droid codebase exploration in this repo, prefer the `codebase-memory-mcp`
MCP server. Do not use the Droid Lode skill unless the task explicitly asks to
test, debug, or compare Lode itself.

## Setup

```bash
cargo build --release --bins
```

Run the daemon stack:

```bash
docker compose up -d --build
```

## Validation

Run these before committing code changes:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Optional blast-radius precision/recall eval:

```bash
python3 scripts/blast_radius_eval.py
```

## Conventions

- Keep Lode local-first. Do not add hosted telemetry or external services by default.
- Prefer JSON output for agent-facing commands and include path/line citations.
- `unsafe_code = "forbid"` is enforced in `Cargo.toml`.
- Keep public CLI and daemon behavior backward compatible unless the change is explicit.
- New maintenance comments must link an issue, for example `TODO(#123)`.

## Security

- Never commit secrets. Keep `.env` files local and use `.env.example` for templates.
- Redact tokens, passwords, and keys before logging.
- Treat repository source as sensitive. Lode should not upload indexed content by default.

## Release

GitHub Releases are handled by `.github/workflows/publish.yml` for `v*` tags or
manual dispatch. Update `Cargo.toml` and `Cargo.lock` when bumping versions.
