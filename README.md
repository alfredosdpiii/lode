# Lode

**A fast, local-first repository knowledge graph for coding agents — written in Rust.**

Lode indexes your codebase into a single bundled SQLite + FTS5 database so agents (and humans) can answer *"what should I read?"*, *"where is this symbol?"*, *"what calls this?"*, and *"what breaks if I change it?"* in milliseconds — with zero hosted index, zero API keys, and zero external dependencies.

---

## See it in action

### 1. Index & build a token-budgeted context pack (`lode index` + `lode context`)
Respects `.gitignore` / `.git/info/exclude` automatically and returns bounded JSON with exact `path:line` citations in ~3 ms.

![Lode index and context demo](docs/assets/lode-context-demo.gif)

### 2. Receiver-aware blast radius & entrypoint impact (`lode impact`)
Resolves constructor assignments, type annotations, `self.<attr>` bindings, and inheritance — while filtering unrelated named imports and rejecting third-party false positives.

![Lode impact and blast radius demo](docs/assets/lode-impact-demo.gif)

---

## How it works

```text
agent / developer
       │
   lode CLI  (or loded local HTTP daemon on 127.0.0.1:7979)
       │
       ├── Walker       .gitignore + .git/info/exclude + generated-file filter
       ├── Parsers      Pure-Rust Python AST (ruff_python_parser), TS/JS, Markdown, TOML/YAML/JSON
       ├── Resolver     Imports, aliases, star imports, receiver types, self.<attr>, EXTENDS inheritance
       ├── Context      Weighted BM25 + identifier IDF + graph edges + directory proximity + path diversity
       └── Blast Radius Symbol-level BFS (CALLS/EXTENDS/HANDLES/CONTAINS) + hop-1 import-filtered files
       │
  ~/.local/share/lode/lode.sqlite   (bundled SQLite + FTS5)
```

### Why it's fast and accurate
- **Zero-infra local binary**: Compiles to two standalone native binaries (`lode` and `loded`) with SQLite bundled in. No Python venv, no Docker requirement, no GPU or embedding model needed.
- **Retrieval (`lode context`)**: Combines weighted SQLite FTS5 BM25, identifier subtokenization (`snake_case` / `camelCase`), candidate-pool IDF specificity, graph proximity from `--from-path`, directory proximity, and path diversification.
- **Impact & blast radius (`lode impact`)**: Tracks local variable assignments (`svc = UserService()`), parameter & variable type annotations (`Optional[T]`, `T | None`, `Annotated[T, ...]`), `with Client() as c:`, class `self.<attr>` bindings, and inherited methods up to 4 `EXTENDS` hops. Never falls back to repo-wide bare names on unknown dotted receivers (preventing `conn.execute()` false positives), and scopes `IMPORTS` expansion to hop 1 only for files that actually import the target symbol/class/module.

---

## Install & Build

Requires **Rust 1.85+**:

```bash
# Install `lode` and `loded` onto your PATH
cargo install --path .
lode --help
```

Or build release binaries locally:

```bash
cargo build --release --bins
./target/release/lode --help
```

### Install the coding-agent skill

This repo ships a reusable agent skill at [skills/lode/SKILL.md](skills/lode/SKILL.md) that teaches coding agents when to reach for `lode index`, `lode context`, `lode symbol`, and `lode impact`:

```bash
AGENT_SKILLS_DIR="$HOME/.agents/skills"   # or ~/.factory/skills, ~/.pi/agent/skills, etc.
mkdir -p "$AGENT_SKILLS_DIR/lode"
cp -R skills/lode/. "$AGENT_SKILLS_DIR/lode/"
```

---

## Quick start

```bash
# 1. Index a repository (incremental by content hash; skips .gitignored paths)
lode index ~/Projects/my-repo --json

# 2. Get a token-budgeted context pack for a task
lode context "how does auth token refresh work" --budget 4000 --json

# 3. Check callers, affected files, and Route entrypoints before editing a symbol
lode impact UserService.save --json

# 4. Exact/partial symbol lookup or FTS search
lode symbol UserService --json
lode search "knowledge graph" --json
```

### Local HTTP daemon (`loded`)

```bash
loded --host 127.0.0.1 --port 7979
curl http://127.0.0.1:7979/health
```

Or run the daemon + optional local embeddings stack via Docker Compose:

```bash
docker compose up -d --build
curl http://127.0.0.1:7979/health
```

### CLI Reference

```bash
lode index PATH [--data-dir DIR] [--json]
lode status [--json]
lode search QUERY [--repo PATH] [--limit N] [--json]
lode symbol NAME [--repo PATH] [--limit N] [--json]
lode context QUERY [--repo PATH] [--budget N] [--from-path PATH] [--exclude-path PATH] [--json]
lode impact TARGET [--repo PATH] [--limit N] [--neighbor-limit N] [--depth N] [--max-nodes N] [--direction up|down|both] [--json]
lode neighbors NODE_ID [--json]
lode embed [--limit N] [--url URL] [--model MODEL] [--json]
lode serve --host 127.0.0.1 --port 7979
```

Default storage location (`LODE_DATA_DIR`, `KG_DATA_DIR`, or `~/.local/share/lode/lode.sqlite`).

---

## Benchmarks

### 1. Cross-file retrieval — RepoBench-R Python v1.1 (300/bucket, merged paths, 5 query lines)

| Split | Bucket | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Median ret |
|---|---|---:|---:|---:|---:|---:|---:|
| `cross_file_first` | lt5 | 0.423 | 0.930 | 0.983 | 0.983 | 0.667 | 3.06 ms |
| `cross_file_first` | easy | 0.257 | 0.620 | 0.857 | 0.997 | 0.486 | 3.48 ms |
| `cross_file_first` | hard | 0.213 | 0.423 | 0.563 | 0.837 | 0.383 | 3.58 ms |
| `cross_file_random` | lt5 | 0.580 | 0.953 | 0.987 | 0.987 | 0.767 | 3.06 ms |
| `cross_file_random` | easy | 0.413 | 0.750 | 0.920 | 0.990 | 0.609 | 3.13 ms |
| `cross_file_random` | hard | 0.280 | 0.547 | 0.690 | 0.897 | 0.459 | 3.43 ms |

> Directional comparison against Table 2 of the [RepoBench-R](https://openreview.net/forum?id=pPjZIOuQuF) paper (*Random, Edit, CodeBERT, Jaccard, UniXcoder*): Lode outperforms UniXcoder on Easy XF-R, Hard XF-F, Hard XF-R, and Easy XF-F Hit@3 (and effectively ties Easy XF-F Hit@1 at 25.7% vs. 25.9%) — using pure lexical + graph ranking in ~3 ms with no GPU. *(Note: RepoBench paper baselines are snippet-only while Lode uses repo-level imports/graph context, and these numbers are from the 300/bucket split.)*

### 2. Blast radius & scale

- **Ground-truth precision/recall suite** (`python3 scripts/blast_radius_eval.py`): **12/12 cases passed** — **1.000 Precision / 1.000 Recall / 1.000 F1** on both receiver-disambiguated call resolution and selective-import blast radius.
- **Large real-world repo stress test** (`44,341` raw files / `7,190` Python files / `~400k` nodes / `~752k` edges):
  - `.gitignore` filtering skipped **35,244** ignored files (**−79.5%**).
  - Hop-1 import filtering cut `conservative_file` blast-radius noise from **139.6 → 3.2** per symbol (**−97.7%**), bringing mean blast radius from **145.8 → 11.5 files**.

---

## Code Hygiene & Quality Guardrails

Lode enforces strict Rust hygiene in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) and pre-commit ([.pre-commit-config.yaml](.pre-commit-config.yaml)):

```bash
# Formatting
cargo fmt --check

# Zero-warning lints (clippy::all = warn, denied in CI)
cargo clippy --all-targets -- -D warnings

# Full unit, integration, graph, storage, and CLI/daemon contract suite (96 tests)
cargo test

# Blast-radius precision/recall + real-repo audit
python3 scripts/blast_radius_eval.py
```

### Built-in invariants & nits
- **`unsafe_code = "forbid"`** is enforced at the crate root in [Cargo.toml](Cargo.toml).
- **Local-first by default**: No telemetry, no hosted APIs, and `lode embed` refuses non-local endpoints (`127.0.0.1`, `localhost`, `::1`, or Docker `embeddings` host only).
- **Deterministic & bounded output**: JSON responses sort keys deterministically, cap traversal with `max_nodes` + an explicit `truncated` flag, and redact sensitive fields in logs.
- **Optional CPU reranker (`--features embed`)**: Pure-Rust `candle` + `tokenizers` code-embedding reranker, off by default so the standard binary stays small and fast.
- **Linked maintenance comments**: Any new `TODO` must reference an issue (e.g. `TODO(#123)`).

---

## License

Apache-2.0.
