#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, TypeVar

from lode.cli import embedding_text
from lode.config import kuzu_path, sqlite_path
from lode.context import build_context_pack
from lode.indexer import index_repo
from lode.storage import (
    connect,
    embedding_counts,
    find_symbol,
    get_neighbors,
    pending_embedding_nodes,
    search_nodes,
    upsert_embedding,
)

ROOT = Path(__file__).resolve().parents[1]

T = TypeVar("T")

DEFAULT_QUERIES = [
    "knowledge graph",
    "build context pack",
    "index repository",
    "embedding queue",
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args)
    except (
        FileNotFoundError,
        RuntimeError,
        sqlite3.Error,
        urllib.error.URLError,
    ) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_summary(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Lode indexing, search, context, graph, and embedding paths."
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository to index")
    parser.add_argument("--data-dir", type=Path, help="Benchmark data directory")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete --data-dir before running. Only applies when --data-dir is set.",
    )
    parser.add_argument(
        "--keep-data-dir",
        action="store_true",
        help="Keep the temporary data directory when --data-dir is omitted.",
    )
    parser.add_argument("--query", action="append", dest="queries", help="Search query")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Symbol lookup target. Defaults to the first indexed local symbol.",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Timing repetitions")
    parser.add_argument("--limit", type=int, default=20, help="Search/result limit")
    parser.add_argument("--budget", type=int, default=4000, help="Context-pack budget")
    parser.add_argument(
        "--include-kuzu",
        action="store_true",
        help="Benchmark SQLite to embedded Kuzu projection",
    )
    parser.add_argument("--embed-url", help="TEI-compatible /embed base URL")
    parser.add_argument("--embed-limit", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Write JSON result to this file")
    parser.add_argument("--json", action="store_true")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Repository does not exist: {repo}")
    repeat = max(1, args.repeat)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.data_dir:
        data_dir = args.data_dir.expanduser().resolve()
        if args.reset and data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="lode-bench-")
        data_dir = Path(temp_dir.name)

    try:
        cold_stats, cold_index_ms = timed(lambda: index_repo(repo, sqlite_path(data_dir)))
        hot_stats, hot_index_ms = timed(lambda: index_repo(repo, sqlite_path(data_dir)))

        with closing(connect(sqlite_path(data_dir))) as conn:
            queries = args.queries or DEFAULT_QUERIES
            symbols = args.symbols or [choose_symbol(conn)]
            symbols = [symbol for symbol in symbols if symbol]
            neighbor_node_id = choose_neighbor_node(conn)
            result: dict[str, Any] = {
                "ok": True,
                "repo": str(repo),
                "data_dir": str(data_dir),
                "cold_index": {
                    "timing_ms": round(cold_index_ms, 3),
                    "stats": asdict(cold_stats),
                },
                "hot_index": {
                    "timing_ms": round(hot_index_ms, 3),
                    "stats": asdict(hot_stats),
                },
                "database": database_metrics(conn, data_dir),
                "search": benchmark_search(conn, queries, repeat, args.limit),
                "symbols": benchmark_symbols(conn, symbols, repeat, args.limit),
                "context": benchmark_context(
                    conn, queries, repeat, args.budget, min(args.limit, 10)
                ),
                "neighbors": benchmark_neighbors(conn, neighbor_node_id, repeat),
            }

            if args.include_kuzu:
                result["kuzu"] = benchmark_kuzu(conn, data_dir)
            if args.embed_url:
                result["embeddings"] = benchmark_embeddings(conn, args.embed_url, args.embed_limit)
                result["database_after_embeddings"] = database_metrics(conn, data_dir)
        return result
    finally:
        if temp_dir and not args.keep_data_dir:
            temp_dir.cleanup()


def benchmark_search(
    conn: sqlite3.Connection, queries: list[str], repeat: int, limit: int
) -> dict[str, Any]:
    out = {}
    for query in queries:
        last_results: list[dict[str, Any]] = []
        timings = []
        for _ in range(repeat):
            current_query = query

            def run_search() -> list[dict[str, Any]]:
                return search_nodes(conn, current_query, limit=limit)

            last_results, elapsed = timed(run_search)
            timings.append(elapsed)
        out[query] = {
            "timing_ms": summarize_timings(timings),
            "last_result_count": len(last_results),
            "top_path": last_results[0]["path"] if last_results else None,
        }
    return out


def benchmark_symbols(
    conn: sqlite3.Connection, symbols: list[str], repeat: int, limit: int
) -> dict[str, Any]:
    out = {}
    for symbol in symbols:
        last_results: list[dict[str, Any]] = []
        timings = []
        for _ in range(repeat):
            current_symbol = symbol

            def run_symbol_lookup() -> list[dict[str, Any]]:
                return find_symbol(conn, current_symbol, limit=limit)

            last_results, elapsed = timed(run_symbol_lookup)
            timings.append(elapsed)
        out[symbol] = {
            "timing_ms": summarize_timings(timings),
            "last_result_count": len(last_results),
            "top_path": last_results[0]["path"] if last_results else None,
        }
    return out


def benchmark_context(
    conn: sqlite3.Connection,
    queries: list[str],
    repeat: int,
    budget: int,
    limit: int,
) -> dict[str, Any]:
    out = {}
    for query in queries:
        last_pack: dict[str, Any] = {}
        timings = []
        for _ in range(repeat):
            current_query = query

            def run_context() -> dict[str, Any]:
                return build_context_pack(conn, current_query, budget=budget, limit=limit)

            last_pack, elapsed = timed(run_context)
            timings.append(elapsed)
        out[query] = {
            "timing_ms": summarize_timings(timings),
            "must_read_count": len(last_pack.get("must_read") or []),
            "top_hit_count": len(last_pack.get("top_hits") or []),
            "confidence": last_pack.get("confidence"),
        }
    return out


def benchmark_neighbors(
    conn: sqlite3.Connection, node_id: str | None, repeat: int
) -> dict[str, Any]:
    if not node_id:
        return {"skipped": True, "reason": "no indexed edges"}
    last_neighbors: dict[str, Any] = {}
    timings = []
    for _ in range(repeat):
        last_neighbors, elapsed = timed(lambda: get_neighbors(conn, node_id))
        timings.append(elapsed)
    return {
        "node_id": node_id,
        "timing_ms": summarize_timings(timings),
        "incoming": len(last_neighbors.get("incoming") or []),
        "outgoing": len(last_neighbors.get("outgoing") or []),
    }


def benchmark_kuzu(conn: sqlite3.Connection, data_dir: Path) -> dict[str, Any]:
    from lode.kuzu_store import sync_from_sqlite

    payload, elapsed = timed(lambda: sync_from_sqlite(conn, kuzu_path(data_dir)))
    return {"timing_ms": round(elapsed, 3), **payload}


def benchmark_embeddings(
    conn: sqlite3.Connection, embed_url: str, embed_limit: int
) -> dict[str, Any]:
    from lode.embeddings import embed_texts, embeddings_model

    nodes = pending_embedding_nodes(conn, limit=embed_limit)
    if not nodes:
        return {"skipped": True, "reason": "no pending embedding nodes"}
    texts = [embedding_text(node) for node in nodes]
    vectors, elapsed = timed(lambda: embed_texts(texts, url=embed_url))
    if len(vectors) != len(nodes):
        raise RuntimeError(
            f"Embedding endpoint returned {len(vectors)} vectors for {len(nodes)} texts"
        )
    model = embeddings_model()
    for node, vector in zip(nodes, vectors):
        upsert_embedding(conn, node["id"], node["repo_id"], vector, model)
    dims = len(vectors[0]) if vectors else 0
    per_second = 1000.0 * len(vectors) / elapsed if elapsed > 0 else 0.0
    return {
        "timing_ms": round(elapsed, 3),
        "embedded": len(vectors),
        "dims": dims,
        "vectors_per_second": round(per_second, 3),
        "model": model,
        **embedding_counts(conn),
    }


def choose_symbol(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT name FROM nodes
        WHERE kind IN ('Function', 'Method', 'Class', 'Route') AND name != ''
        ORDER BY CASE kind WHEN 'Function' THEN 0 WHEN 'Method' THEN 1 ELSE 2 END,
                 length(name), name
        LIMIT 1
        """
    ).fetchone()
    return str(row["name"]) if row else ""


def choose_neighbor_node(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT src AS node_id FROM edges
        GROUP BY src
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row["node_id"]) if row else None


def database_metrics(conn: sqlite3.Connection, data_dir: Path) -> dict[str, Any]:
    counts = {}
    for table in ["repos", "files", "nodes", "edges", "embedding_queue", "embeddings"]:
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return {
        "counts": counts,
        "sqlite_bytes": file_size(sqlite_path(data_dir)),
        "sqlite_wal_bytes": file_size(sqlite_path(data_dir).with_suffix(".sqlite-wal")),
        "sqlite_shm_bytes": file_size(sqlite_path(data_dir).with_suffix(".sqlite-shm")),
        "kuzu_bytes": directory_size(kuzu_path(data_dir)),
    }


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def timed(func: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter_ns()
    value = func()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return value, elapsed_ms


def summarize_timings(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "p50": round(median(values), 3),
        "mean": round(mean(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def print_summary(result: dict[str, Any]) -> None:
    print("Lode operational benchmark")
    print(f"repo: {result['repo']}")
    print(f"data_dir: {result['data_dir']}")
    print(f"cold index: {result['cold_index']['timing_ms']:.3f} ms")
    print(f"hot index:  {result['hot_index']['timing_ms']:.3f} ms")
    counts = result["database"]["counts"]
    print(
        "db: "
        f"files={counts['files']} nodes={counts['nodes']} edges={counts['edges']} "
        f"queued={counts['embedding_queue']} embedded={counts['embeddings']}"
    )
    print("search p50 ms:")
    for query, payload in result["search"].items():
        print(f"  {query!r}: {payload['timing_ms']['p50']:.3f}")
    print("context p50 ms:")
    for query, payload in result["context"].items():
        print(f"  {query!r}: {payload['timing_ms']['p50']:.3f}")
    if "kuzu" in result:
        print(f"kuzu sync: {result['kuzu']['timing_ms']:.3f} ms")
    if "embeddings" in result:
        print(f"embeddings: {json.dumps(result['embeddings'], sort_keys=True)}")


if __name__ == "__main__":
    raise SystemExit(main())
