"""Diff the Rust `lode` port against the Python implementation.

Indexes the lode repo and the first N samples per bucket of RepoBench-R
cross_file_first with both implementations, then compares node/edge sets in
each SQLite DB and the paths returned by `context`. Writes a JSON summary to
/tmp/lode-rust-diff-summary.json.

Usage (from the repo root):
    uv run python rust/scripts/diff_against_python.py [--per-bucket 30] [--rust-bin PATH]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from repobench_eval import (
    BENCH_DIR,
    REPO,
    RustError,
    find_rust_bin,
    pack_paths,
    prepare_sample,
    python_context,
    rust_context,
    rust_index,
    select_samples,
)

from benchmarks.repobench_adapter import query_from_sample
from lode.config import sqlite_path
from lode.indexer import index_repo

SUMMARY_PATH = Path("/tmp/lode-rust-diff-summary.json")
LODE_QUERIES = [
    "build_context_pack",
    "resolve graph imports",
    "blast radius",
    "fts query tokens",
    "daemon impact route",
    "embedding queue",
    "parse python ast",
    "repo_filter unsafe path",
    "kuzu sync",
    "markdown headings",
]
MAX_DETAILED_REPOS = 3
MAX_EXAMPLES = 5


def jaccard(a: set[Any] | list[Any], b: set[Any] | list[Any]) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def load_graph(db: Path) -> dict[str, set[Any]]:
    with closing(sqlite3.connect(db)) as conn:
        nodes = conn.execute(
            "SELECT id, kind, qname, path, start_line, end_line FROM nodes"
        ).fetchall()
        edges = conn.execute("SELECT src, dst, kind, confidence, detail FROM edges").fetchall()
    by_id = {row[0]: row[1:] for row in nodes}

    def strict(node_id: str) -> Any:
        return by_id.get(node_id, ("<unresolved>", node_id))

    def loose(node_id: str) -> Any:
        node = by_id.get(node_id)
        return (node[0], node[1], node[2]) if node else ("<unresolved>", node_id)

    return {
        "nodes": {row[1:] for row in nodes},
        "node_ids": set(by_id),
        "edges": {(strict(s), strict(d), k, c, det) for s, d, k, c, det in edges},
        "edges_loose": {(loose(s), loose(d)[:2], k) for s, d, k, _, _ in edges},
    }


def compare_sets(name: str, py: set[Any], rs: set[Any], verbose: bool) -> dict[str, Any]:
    inter = py & rs
    out = {
        "py": len(py),
        "rs": len(rs),
        "inter": len(inter),
        "jaccard": jaccard(py, rs),
        "exact": py == rs,
    }
    print(
        f"    {name:12s} py={len(py):6d} rs={len(rs):6d} inter={len(inter):6d} J={out['jaccard']:.4f}"
    )
    if verbose and py != rs:
        for label, only in (("py-only", py - rs), ("rs-only", rs - py)):
            for item in sorted(only, key=repr)[:MAX_EXAMPLES]:
                print(f"      {label}: {item!r}"[:300])
    return out


def compare_context(py_paths: list[str], rs_paths: list[str]) -> dict[str, Any]:
    return {
        "top1": py_paths[:1] == rs_paths[:1],
        "top5": py_paths[:5] == rs_paths[:5],
        "top10_jaccard": jaccard(py_paths[:10], rs_paths[:10]),
    }


def new_result(label: str, errors: list[str]) -> dict[str, Any]:
    return {"label": label, "context": [], "py_ctx_ms": [], "rs_ctx_ms": [], "errors": errors}


def diff_repo(
    label: str,
    repo_dir: Path,
    queries: list[tuple[str, dict[str, Any]]],
    rust_bin: Path,
    work: Path,
    verbose: bool,
) -> dict[str, Any]:
    py_data, rs_data = work / "py-data", work / "rs-data"
    py_data.mkdir(parents=True)
    rs_data.mkdir(parents=True)
    result = new_result(label, [])

    t0 = time.perf_counter_ns()
    index_repo(repo_dir, sqlite_path(py_data))
    result["py_index_ms"] = (time.perf_counter_ns() - t0) / 1_000_000
    try:
        _, result["rs_index_ms"] = rust_index(rust_bin, repo_dir, rs_data)
    except RustError as exc:
        result["errors"].append(str(exc))
        print(f"  {label}: rust index failed: {exc}")
        return result

    print(f"  {label}: index py={result['py_index_ms']:.0f}ms rs={result['rs_index_ms']:.0f}ms")
    py_g, rs_g = load_graph(sqlite_path(py_data)), load_graph(sqlite_path(rs_data))
    for key in ("nodes", "node_ids", "edges", "edges_loose"):
        result[key] = compare_sets(key, py_g[key], rs_g[key], verbose)

    for query, kwargs in queries:
        py_pack, py_ms = python_context(py_data, query, **kwargs)
        try:
            rs_pack, rs_ms = rust_context(rust_bin, rs_data, query, **kwargs)
        except RustError as exc:
            result["errors"].append(f"context {query[:40]!r}: {exc}")
            continue
        py_paths, rs_paths = pack_paths(py_pack), pack_paths(rs_pack)
        cmp = compare_context(py_paths, rs_paths)
        result["context"].append(cmp)
        result["py_ctx_ms"].append(py_ms)
        result["rs_ctx_ms"].append(rs_ms)
        if verbose and not cmp["top5"]:
            print(f"    ctx {query[:50]!r}\n      py={py_paths[:5]}\n      rs={rs_paths[:5]}")
    return result


def timing(values: list[float]) -> dict[str, float]:
    return {"median_ms": statistics.median(values), "total_ms": sum(values)} if values else {}


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = [r for r in results if "nodes" in r]
    ctx = [c for r in results for c in r["context"]]
    summary: dict[str, Any] = {
        "repos": len(results),
        "repos_indexed_by_both": len(indexed),
        "errors": sum(len(r["errors"]) for r in results),
        "context_queries": len(ctx),
    }
    for key in ("nodes", "node_ids", "edges", "edges_loose"):
        summary[f"{key}_jaccard_mean"] = mean([r[key]["jaccard"] for r in indexed])
        summary[f"{key}_exact_rate"] = mean([float(r[key]["exact"]) for r in indexed])
    summary["context_top1_rate"] = mean([float(c["top1"]) for c in ctx])
    summary["context_top5_rate"] = mean([float(c["top5"]) for c in ctx])
    summary["context_top10_jaccard_mean"] = mean([c["top10_jaccard"] for c in ctx])
    summary["index_python"] = timing([r["py_index_ms"] for r in results if "py_index_ms" in r])
    summary["index_rust"] = timing([r["rs_index_ms"] for r in indexed])
    summary["context_python"] = timing([ms for r in results for ms in r["py_ctx_ms"]])
    summary["context_rust_incl_startup"] = timing([ms for r in results for ms in r["rs_ctx_ms"]])
    summary["error_examples"] = [e for r in results for e in r["errors"]][:10]
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n==================== SUMMARY ====================")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:32s} {value:.4f}")
        elif isinstance(value, dict):
            pretty = ", ".join(f"{k}={v:.1f}" for k, v in value.items()) or "n/a"
            print(f"  {key:32s} {pretty}")
        elif isinstance(value, list):
            for line in value:
                print(f"  error: {line}"[:300])
        else:
            print(f"  {key:32s} {value}")
    print(f"  (JSON written to {SUMMARY_PATH})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-bucket", type=int, default=30)
    parser.add_argument("--rust-bin", default=None)
    args = parser.parse_args()
    rust_bin = find_rust_bin(args.rust_bin)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lode-rs-diff-") as tmp:
        work = Path(tmp)
        lode_queries: list[tuple[str, dict[str, Any]]] = [(q, {}) for q in LODE_QUERIES]
        print("== lode repo ==")
        results.append(diff_repo("lode", REPO, lode_queries, rust_bin, work / "lode", verbose=True))

        print(f"== RepoBench cross_file_first (per_bucket={args.per_bucket}) ==")
        mismatches_shown = 0
        for idx, (line_no, sample) in enumerate(
            select_samples(BENCH_DIR / "cross_file_first.jsonl", args.per_bucket)
        ):
            sdir = work / f"s{idx}"
            try:
                mat = prepare_sample(sample, sdir / "repo")
            except Exception as exc:
                results.append(new_result(f"line{line_no}", [f"materialize: {exc}"]))
                continue
            kwargs = {
                "from_path": mat.current_path,
                "exclude": {mat.current_path},
                "include_related": False,
            }
            queries = [(query_from_sample(sample, 5), kwargs)]
            verbose = mismatches_shown < MAX_DETAILED_REPOS
            res = diff_repo(
                f"line{line_no}", sdir / "repo", queries, rust_bin, sdir, verbose=verbose
            )
            graph_differs = "nodes" in res and not (res["nodes"]["exact"] and res["edges"]["exact"])
            if verbose and (graph_differs or any(not c["top5"] for c in res["context"])):
                mismatches_shown += 1
            results.append(res)

    summary = summarize(results)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
