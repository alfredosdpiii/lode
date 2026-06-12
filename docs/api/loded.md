# loded API

Generated from `scripts/generate_openapi_docs.py`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Daemon health check. |
| `GET` | `/metrics` | Prometheus text metrics for the local daemon. |
| `GET` | `/status` | Indexed repository status. |
| `GET` | `/search` | Search indexed code and docs. |
| `GET` | `/impact` | Impact report for a symbol or node. |
| `POST` | `/index` | Index a repository path. |
| `POST` | `/search` | Search indexed code and docs. |
| `POST` | `/context` | Build a bounded agent context pack. |
| `POST` | `/impact` | Impact report for a symbol or node. |
