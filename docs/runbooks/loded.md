# loded Runbook

## Health

```bash
curl -sS http://127.0.0.1:7979/health
curl -sS http://127.0.0.1:7979/metrics
```

## Local restart

```bash
docker compose down
docker compose up -d --build
```

## Reindex a repository

```bash
curl -sS -X POST http://127.0.0.1:7979/index \
  -H 'content-type: application/json' \
  -d '{"path":"/repos/lode"}'
```

## Release rollback

Lode publishes on `v*` tags. To roll back a bad release, yank or delete the PyPI
release if appropriate, then publish a new patch version from the last known good
commit. Do not force-push tags that may already be consumed by users.
