# Rust port of Lode

`rust/` is a Rust port of `src/lode`. It ships two binaries, `lode` and `loded`. They read and
write the same SQLite schema, at the same location, as the Python package, so either
implementation can use a database the other built.

## Layout

| Rust module | Python source | Notes |
|---|---|---|
| `config.rs`, `features.rs`, `observability.rs`, `model.rs` | same names | Same data dir rules (`LODE_DATA_DIR`, `KG_DATA_DIR`, `~/.local/share/lode`), node ids and feature flags. |
| `storage.rs` | `storage.py` | rusqlite with `bundled` SQLite. Same schema and FTS5 table. `bm25(node_fts, 0.0, 8.0, 12.0, 4.0, 1.5, 6.0)`: the first weight belongs to the UNINDEXED `node_id` column. |
| `pyast.rs`, `indexer.rs` | `indexer.py` | Python is parsed with `ruff_python_parser` and `ruff_python_ast` (git dependency, tag `0.17.6`). TS/JS, Markdown and config files use the same regex heuristics as `indexer.py`. Parsing runs on rayon threads instead of a process pool. |
| `graph.rs` | `graph.py` | `resolve_graph`, `blast_radius`, `impact_targets`, `impact_report`. |
| `context.rs` | `context.py` | Ranks exactly as the current Python code does: weighted BM25, identifier specificity with candidate-pool IDF, graph boosts from `from_path`, directory proximity, and path diversification. |
| `embeddings.rs` | `embeddings.py` | Local endpoints only (loopback or the Docker host `embeddings`). Uses plain HTTP over `std::net`. |
| `cli.rs`, `bin/lode.rs` | `cli.py` | clap. Same subcommands, flags, JSON keys and exit codes. |
| `daemon.rs`, `bin/loded.rs` | `daemon.py` | tiny_http on `127.0.0.1:7979` by default. Same routes, status codes, JSON keys, `x-request-id` header and `/metrics` output. |
| `kuzu.rs` | `kuzu_store.py` | Not ported; see below. |
| `json.rs`, `text.rs` | (stdlib behavior) | `json.dumps(sort_keys=True, indent=2)` output, `str.splitlines`, the camelCase splitting regex, and `repr` length for the context cost estimate. |

## Decisions and differences

- **Python parser.** The port uses ruff's parser, which is pure Rust; tree-sitter was removed.
  CPython rejects some files that ruff accepts and the other way round, and the two report
  errors at different places. For a file that does not parse, `pyast.rs` turns ruff's error
  into CPython's line and message for the cases RepoBench repositories actually hit:
  unexpected indent, a missing indented block after a header (including headers that span
  several lines), `try` without `except`/`finally`, and an unclosed bracket. Any other error
  gets CPython's generic `invalid syntax` message, at ruff's line. The `ParseError` node id
  depends on that line, which is why the mapping matters.
- **Calls in annotations.** CPython's `ast` walk in `iter_imports_and_calls` treats `ast.arg`
  as a leaf, so calls inside parameter annotations are skipped. The Rust visitor skips them
  too.
- **Bug kept for parity: fallback imports.** When a Python file does not parse,
  `indexer.py` creates `ExternalDependency` nodes for its imports, but appends them to a list
  that is then thrown away. Only the `IMPORTS` edges are kept, and they point at node ids that
  are never stored. The Rust port does the same (`parse_python` drops external nodes on the
  fallback path). Fixing it means changing both implementations together.
- **Kuzu.** Optional in Python, and there is no Kuzu crate in Rust. `lode kuzu-sync` and
  `lode index --sync-kuzu` return the error "Kuzu is not installed …" (exit code 1, or
  `{"ok": false, ...}` with `--json`). The `kuzu` cargo feature is off by default; turning it
  on only changes the message.
- **`embed` cargo feature (not in Python, off by default).** Adds a local code-embedding
  reranker (`src/rerank.rs`): jinaai/jina-embeddings-v2-base-code run on the CPU with candle,
  `tokenizers` with pure-Rust regex. No C code beyond the bundled SQLite. `lode rerank-model
  --download` is the only command that uses the network; `lode context --rerank` errors if the
  model is missing. Embeddings match the reference implementation at cosine 1.000. On RepoBench
  (merged paths, 100 per bucket) the best fusion only tied the baseline and embeddings alone were
  worse in every bucket, at 2.6-5.2 s per query, so it is not used by default. The `lode embed`
  command, which talks to a local embeddings service, is always built, as in Python.
- **Embeddings over HTTPS.** Not supported. The endpoint must be local anyway, so the port
  speaks plain HTTP and rejects `https://` with a clear error.
- **Extra context flags.** `lode context` also accepts `--from-path`, `--exclude-path`
  (repeatable) and `--no-related`. In Python these options exist only in the library
  (`build_context_pack`); the CLI flags let the eval scripts drive the Rust binary. Nothing
  that already works changes.
- **File walk order.** Files are walked in sorted order rather than `os.walk` directory
  order. Stored rows are the same; only the order of insertion differs.

## Verification

Run from the repo root on 2026-11-21:

- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and `cargo test` all pass.
  Behaviour tests ported from `tests/test_lode.py`, `test_graph.py`,
  `test_storage_neighbors.py` and `test_public_compatibility.py` are in `rust/tests/`; each
  file lists the Python tests it skips and why.
- `uv run python rust/scripts/diff_against_python.py --per-bucket 30` covers the lode repo
  and 90 RepoBench `cross_file_first` repos (91 repos, 100 context queries, 0 errors). It
  compares node and edge sets in each database: Jaccard 1.0000 and an exact-match rate of
  1.0000 for nodes, node ids, edges and loose edges. Context agreement is 1.0000 for top-1
  and top-5, and the top-10 Jaccard is 1.0000. On the lode repo itself, both
  implementations store 2866 nodes, 5764 edges, 923 FTS rows and 636 queued embeddings,
  identically.
- `uv run python rust/scripts/repobench_eval.py 50 5 --python` runs the same sampling and
  metrics as `/tmp/lode_hillclimb_eval.py`. Rust and Python give identical numbers, and both
  match `/tmp/lode_eval_iter5.txt`:

| split | bucket | H@1 | H@3 | H@5 | H@10 | MRR | Rust ret (with process start) | Python ret (in process) |
|---|---|---|---|---|---|---|---|---|
| cross_file_first | lt5 | 0.440 | 0.840 | 0.920 | 0.920 | 0.647 | 3.50 ms | 2.16 ms |
| cross_file_first | easy | 0.300 | 0.640 | 0.860 | 0.920 | 0.495 | 3.48 ms | 2.35 ms |
| cross_file_first | hard | 0.120 | 0.480 | 0.640 | 0.860 | 0.347 | 6.63 ms | 4.23 ms |
| cross_file_random | lt5 | 0.440 | 0.960 | 0.960 | 0.960 | 0.690 | 3.48 ms | 2.14 ms |
| cross_file_random | easy | 0.380 | 0.820 | 0.960 | 0.980 | 0.615 | 2.97 ms | 1.96 ms |
| cross_file_random | hard | 0.260 | 0.460 | 0.580 | 0.860 | 0.420 | 3.52 ms | 2.68 ms |

Timings from the diff run (91 repos): indexing median 7.4 ms in Rust and 10.2 ms in
Python (837 ms and 1637 ms in total). Context median 3.6 ms in Rust, including about 1 ms
to start a process per query, against 2.4 ms for Python in process. On the lode repo,
`lode index` takes 0.10 s in Rust and 0.32 s for `python -m lode`.
