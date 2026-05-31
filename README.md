# Lode

Lode is a fully local repository knowledge graph for coding agents.

It is a CLI-first, Docker-runnable code intelligence service. Lode indexes local repositories into a fast SQLite hot path and projects the same facts into embedded Kuzu for graph traversal. It is designed for agents that need quick answers to: where is this symbol, what is connected to it, what should I read, and what breaks if I change it?

## Goals

- Fully local: no accounts, no hosted control plane, no remote API calls by default.
- Agent-native: compact JSON, bounded output, confidence labels, and file:line citations.
- CLI first: agents call `lode` directly; MCP can be a thin compatibility shim later.
- Hybrid storage: SQLite for exact/FTS hot-path lookup, Kuzu for graph/Cypher/vector workloads.
- Docker on login: run `loded` as a local service and keep indexes fresh.

## Current state

This is an early open-source MVP. It currently provides:

- `lode index PATH` for Python, TypeScript/JavaScript, Markdown, and config-ish files.
- `lode search QUERY --json` over SQLite FTS5.
- `lode symbol NAME --json` for exact-ish symbol lookup.
- `lode context QUERY --json --budget N` for an agent context pack.
- `lode neighbors NODE_ID --json` for direct graph neighbors.
- `loded` local HTTP daemon with `/health`, `/status`, `/index`, `/search`, and `/context`.
- Optional Kuzu projection code when the `kuzu` extra is installed.
- Docker Compose with a local TEI embeddings service using `Snowflake/snowflake-arctic-embed-s`.

## Quick start

```bash
cd ~/Projects/lode
uv run lode index ~/Projects/lode
uv run lode search "knowledge graph" --json
uv run lode context "how does indexing work" --json --budget 4000
```

Run the local daemon:

```bash
uv run loded --host 127.0.0.1 --port 7979
```

Use Docker Compose:

```bash
docker compose up -d --build
curl http://127.0.0.1:7979/health
```

The `loded` container runs as `${LODE_UID:-1000}:${LODE_GID:-1000}` so its SQLite file stays writable by the host CLI. Export `LODE_UID=$(id -u)` and `LODE_GID=$(id -g)` first if your user is not UID/GID 1000.

Index a mounted repo through the daemon:

```bash
curl -sS -X POST http://127.0.0.1:7979/index \
  -H 'content-type: application/json' \
  -d '{"path":"/repos/lode"}' | jq
```

## Architecture

```text
agent / human
    |
  lode CLI
    |
localhost HTTP or direct DB
    |
  loded daemon
    |--------------------------|
    | scanner / parser         |
    | resolver                 |
    | context pack builder     |
    | embedding queue          |
    | Kuzu projector           |
    |--------------------------|
       |                   |
 SQLite hot index       Kuzu graph DB
       |                   |
       +---- fact projection ----+
               |
        TEI embeddings service
```

SQLite is the fast operational index. Kuzu is the graph analytics and Cypher projection. Facts should eventually be append-only and replayable so both projections can be rebuilt.

## CLI commands

```bash
lode index PATH [--data-dir DIR] [--sync-kuzu]
lode status [--json]
lode search QUERY [--repo PATH] [--limit N] [--json]
lode symbol NAME [--repo PATH] [--limit N] [--json]
lode context QUERY [--repo PATH] [--budget N] [--json]
lode neighbors NODE_ID [--json]
lode kuzu-sync
lode embed [--limit N] [--url URL] [--model MODEL] [--json]
lode serve --host 127.0.0.1 --port 7979
```

`kg` and `kgd` are temporary aliases for `lode` and `loded` while the project is young.

## Storage layout

Default data directory:

```text
~/.local/share/lode/
  lode.sqlite
  lode.kuzu/
```

## Embeddings

The Docker Compose file starts Hugging Face Text Embeddings Inference with `Snowflake/snowflake-arctic-embed-s`. It exposes `/embed` on `127.0.0.1:7980` for local smoke tests and wires the daemon with `LODE_EMBEDDINGS_URL=http://embeddings:80`. Embeddings are intentionally secondary to exact search and graph traversal.

Model choice: Exa/web research found `snowflake-arctic-embed-s` is the strongest 33M-parameter / 384-dimension small English retrieval model in its comparison set, with MTEB retrieval NDCG@10 of 51.98 versus 51.68 for `BAAI/bge-small-en-v1.5`. It also has ONNX artifacts and was smoke-tested successfully with TEI CPU `/embed`.

Embed queued nodes after indexing:

```bash
docker compose up -d embeddings
LODE_EMBEDDINGS_URL=http://127.0.0.1:7980 \
  LODE_EMBEDDINGS_MODEL=Snowflake/snowflake-arctic-embed-s \
  uv run lode embed --limit 32 --json
```

The first attempted default, `Qwen/Qwen3-Embedding-0.6B`, is not a safe TEI CPU default here: the container downloads the model, reports missing ONNX artifacts, falls back to Candle CPU warmup, and restarts before `/embed` serves. `BAAI/bge-small-en-v1.5` works, but `Snowflake/snowflake-arctic-embed-s` is the current small-model default because it is the same size class and scored slightly better in the retrieved benchmark data.

## Benchmarks

Lode includes two benchmark entrypoints:

```bash
# Local operational benchmark: cold/hot index, search, symbols, context, graph, optional Kuzu
uv run python scripts/bench_lode.py --repo . --include-kuzu --json

# If TEI is running locally, include embedding throughput/persistence
docker compose up -d embeddings
uv run python scripts/bench_lode.py --repo . --embed-url http://127.0.0.1:7980 --json
```

For RepoBench-style retrieval quality, export a RepoBench split to JSONL and run the adapter:

```bash
# Expected fields match tianyang/repobench_python_v1.1: context, cropped_code, file_path, gold_snippet_index
uv run python benchmarks/repobench_adapter.py --input repobench_cross_file_first.jsonl --limit 100 --json
```

The adapter materializes each sample as a tiny repository and reports `hit_at_k` plus MRR for whether Lode ranks the gold cross-file snippet path. It is intended as a retrieval benchmark, not a code-generation benchmark.

## License

Apache-2.0.
