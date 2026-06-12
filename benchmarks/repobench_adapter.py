#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any

from lode.config import sqlite_path
from lode.context import build_context_pack
from lode.indexer import index_repo
from lode.storage import connect, search_nodes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate(args)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
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
        description=(
            "Evaluate Lode retrieval on RepoBench-style JSONL samples by "
            "materializing each sample as a tiny repository."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL file or dir")
    parser.add_argument("--limit", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--start", type=int, default=0, help="Samples to skip first")
    parser.add_argument(
        "--mode",
        choices=["search", "context", "hybrid"],
        default="hybrid",
        help="Lode retrieval path to score",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--query-lines", type=int, default=12)
    parser.add_argument("--search-limit", type=int, default=20)
    parser.add_argument("--context-budget", type=int, default=4000)
    parser.add_argument("--details", action="store_true", help="Include per-sample rows")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"RepoBench input does not exist: {input_path}")
    top_k = sorted({k for k in args.top_k if k > 0})
    if not top_k:
        raise ValueError("--top-k must contain at least one positive integer")

    rows = []
    errors = []
    reciprocal_ranks = []
    hits = {k: 0 for k in top_k}
    evaluated = 0
    skipped = 0
    index_timings = []
    retrieve_timings = []

    with tempfile.TemporaryDirectory(prefix="lode-repobench-") as temp_root:
        root = Path(temp_root)
        for raw_index, (source, line_no, sample) in enumerate(iter_samples(input_path)):
            if raw_index < args.start:
                continue
            if args.limit is not None and evaluated >= args.limit:
                break
            sample_id = str(sample.get("idx") or f"{source.name}:{line_no}")
            try:
                sample_root = root / f"sample-{evaluated:06d}"
                repo_dir = sample_root / "repo"
                data_dir = sample_root / "data"
                repo_dir.mkdir(parents=True)
                data_dir.mkdir(parents=True)
                target_path = materialize_sample(sample, repo_dir)
                query = query_from_sample(sample, args.query_lines)
                _, index_ms = timed(
                    lambda repo_dir=repo_dir, data_dir=data_dir: index_repo(
                        repo_dir, sqlite_path(data_dir)
                    )
                )
                with closing(connect(sqlite_path(data_dir))) as conn:
                    ranked_paths, retrieve_ms = timed(
                        lambda query=query: retrieve_paths(
                            conn,
                            query,
                            args.mode,
                            args.search_limit,
                            args.context_budget,
                        )
                    )
                rank = first_rank(ranked_paths, target_path)
                evaluated += 1
                index_timings.append(index_ms)
                retrieve_timings.append(retrieve_ms)
                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                    for k in top_k:
                        if rank <= k:
                            hits[k] += 1
                else:
                    reciprocal_ranks.append(0.0)
                if args.details:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "target_path": target_path,
                            "rank": rank,
                            "query": query,
                            "ranked_paths": ranked_paths[: max(top_k)],
                            "index_ms": round(index_ms, 3),
                            "retrieve_ms": round(retrieve_ms, 3),
                        }
                    )
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                OSError,
                sqlite3.Error,
            ) as exc:
                skipped += 1
                errors.append({"sample_id": sample_id, "error": str(exc)})
                if args.fail_fast:
                    raise ValueError(f"failed on {sample_id}: {exc}") from exc
            finally:
                if "sample_root" in locals() and sample_root.exists():
                    shutil.rmtree(sample_root)

    metrics = {f"hit_at_{k}": round(hits[k] / evaluated, 6) if evaluated else 0.0 for k in top_k}
    metrics["mrr"] = round(mean(reciprocal_ranks), 6) if reciprocal_ranks else 0.0
    return {
        "ok": True,
        "input": str(input_path),
        "mode": args.mode,
        "samples_evaluated": evaluated,
        "samples_skipped": skipped,
        "top_k": top_k,
        "metrics": metrics,
        "timing_ms": {
            "index_mean": round(mean(index_timings), 3) if index_timings else 0.0,
            "retrieve_mean": round(mean(retrieve_timings), 3) if retrieve_timings else 0.0,
        },
        "details": rows if args.details else [],
        "errors": errors[:20],
    }


def iter_samples(path: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                if not isinstance(data, dict):
                    raise TypeError(f"{file_path}:{line_no} is not a JSON object")
                yield file_path, line_no, data


def materialize_sample(sample: dict[str, Any], repo_dir: Path) -> str:
    contexts = sample.get("context")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("sample has no context list")
    gold_index = int(sample["gold_snippet_index"])
    if gold_index < 0 or gold_index >= len(contexts):
        raise IndexError("gold_snippet_index is outside context list")

    used_paths: set[str] = set()
    target_path = ""
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise TypeError("context entry is not an object")
        rel = unique_path(
            safe_relative_path(context.get("path"), f"context_{index}.py"),
            used_paths,
            index,
        )
        snippet = str(context.get("snippet") or context.get("content") or "")
        write_file(repo_dir / rel, snippet)
        if index == gold_index:
            target_path = rel.as_posix()

    current_path = unique_path(
        safe_relative_path(sample.get("file_path"), "current.py"),
        used_paths,
        len(contexts),
    )
    cropped_code = str(sample.get("cropped_code") or sample.get("all_code") or "")
    write_file(repo_dir / current_path, cropped_code)
    if not target_path:
        raise ValueError("missing gold target path")
    return target_path


def safe_relative_path(raw: Any, fallback: str) -> Path:
    value = str(raw or fallback).replace("\\", "/").lstrip("/")
    parts = [part for part in PurePosixPath(value).parts if part not in {"", ".", ".."}]
    if not parts:
        parts = [fallback]
    path = Path(*parts)
    if not path.suffix:
        path = path.with_suffix(".py")
    return path


def unique_path(path: Path, used_paths: set[str], index: int) -> Path:
    candidate = path
    while candidate.as_posix() in used_paths:
        candidate = candidate.with_name(f"{candidate.stem}__ctx{index}{candidate.suffix}")
    used_paths.add(candidate.as_posix())
    return candidate


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def query_from_sample(sample: dict[str, Any], query_lines: int) -> str:
    cropped_code = str(sample.get("cropped_code") or "")
    lines = [line.rstrip() for line in cropped_code.splitlines() if line.strip()]
    if not lines:
        return str(sample.get("file_path") or sample.get("repo_name") or "")
    raw_query = "\n".join(lines[-max(1, query_lines) :])
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw_query)
    return " ".join(identifiers[-64:]) if identifiers else raw_query


def retrieve_paths(
    conn: sqlite3.Connection,
    query: str,
    mode: str,
    search_limit: int,
    context_budget: int,
) -> list[str]:
    paths: list[str] = []
    if mode in {"search", "hybrid"}:
        paths.extend(row["path"] for row in search_nodes(conn, query, limit=search_limit))
    if mode in {"context", "hybrid"}:
        pack = build_context_pack(conn, query, budget=context_budget, limit=min(search_limit, 20))
        paths.extend(item["path"] for item in pack.get("top_hits") or [])
        paths.extend(item["path"] for item in pack.get("must_read") or [])
    return unique_preserve_order(paths)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def first_rank(ranked_paths: list[str], target_path: str) -> int | None:
    for index, path in enumerate(ranked_paths, 1):
        if path == target_path:
            return index
    return None


def timed(func):
    start = time.perf_counter_ns()
    value = func()
    return value, (time.perf_counter_ns() - start) / 1_000_000


def print_summary(result: dict[str, Any]) -> None:
    print("RepoBench-style Lode retrieval benchmark")
    print(f"input: {result['input']}")
    print(f"mode: {result['mode']}")
    print(f"samples: evaluated={result['samples_evaluated']} skipped={result['samples_skipped']}")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.6f}")
    print(
        "timing mean ms: "
        f"index={result['timing_ms']['index_mean']:.3f} "
        f"retrieve={result['timing_ms']['retrieve_mean']:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
