from __future__ import annotations

import argparse
import json
import keyword
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any

from lode.config import sqlite_path
from lode.context import build_context_pack
from lode.indexer import index_repo
from lode.storage import connect, search_nodes

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = set(keyword.kwlist) | {
    "self",
    "cls",
    "none",
    "true",
    "false",
    "return",
    "def",
    "class",
    "import",
    "from",
    "as",
    "with",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "super",
    "__init__",
}


@dataclass
class _Accumulator:
    hits: dict[int, int] = field(default_factory=lambda: {})
    reciprocal_ranks: list[float] = field(default_factory=list)
    index_timings: list[float] = field(default_factory=list)
    retrieve_timings: list[float] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    evaluated: int = 0
    skipped: int = 0

    def __post_init__(self) -> None:
        if not self.hits:
            self.hits = {}


def _make_acc(top_k: list[int]) -> _Accumulator:
    return _Accumulator(hits={k: 0 for k in top_k})


@dataclass(slots=True)
class _MaterializedCandidate:
    path: str
    identifier: str
    snippet: str
    identifier_terms: frozenset[str]
    path_terms: frozenset[str]
    snippet_terms: frozenset[str]
    order: int


@dataclass(slots=True)
class _MaterializedSample:
    target_path: str
    current_path: str
    cropped_code: str
    candidates: list[_MaterializedCandidate]


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

    dataset = _resolve_dataset(input_path)
    input_files = _resolve_input_files(input_path)
    split_names = [f.stem for f in input_files]

    # Per-split accumulators
    split_accs: dict[str, _Accumulator] = {}
    for name in split_names:
        split_accs[name] = _make_acc(top_k)

    # Bucket and level accumulators per split
    split_buckets: dict[str, dict[str, _Accumulator]] = {}
    split_levels: dict[str, dict[str, _Accumulator]] = {}
    for name in split_names:
        split_buckets[name] = {
            "lt5_candidates": _make_acc(top_k),
            "easy_5_9_candidates": _make_acc(top_k),
            "hard_10_plus_candidates": _make_acc(top_k),
        }
        split_levels[name] = {}

    details_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    global_evaluated = 0

    with tempfile.TemporaryDirectory(prefix="lode-repobench-") as temp_root:
        root = Path(temp_root)
        sample_iter = iter_samples(input_path)
        raw_index = -1
        while True:
            if args.limit is not None and global_evaluated >= args.limit:
                break
            try:
                source, line_no, sample = next(sample_iter)
                raw_index += 1
            except StopIteration:
                break
            if raw_index < args.start:
                continue
            split_name = source.stem
            split_acc = split_accs[split_name]
            sample_id = f"{source.name}:{line_no}"

            # Pre-classify bucket and level before materialization so skips
            # are still attributed to the correct bucket and level subtotals.
            contexts = sample.get("context")
            context_count = len(contexts) if isinstance(contexts, list) else 0
            if context_count < 5:
                bucket_name = "lt5_candidates"
            elif context_count <= 9:
                bucket_name = "easy_5_9_candidates"
            else:
                bucket_name = "hard_10_plus_candidates"
            bucket_acc = split_buckets[split_name][bucket_name]

            level = sample.get("level")
            level_name = str(level) if isinstance(level, str) else "unknown"
            if level_name not in split_levels[split_name]:
                split_levels[split_name][level_name] = _make_acc(top_k)
            level_acc = split_levels[split_name][level_name]

            try:
                sample_root = root / f"sample-{split_name}-{split_acc.evaluated:06d}"
                repo_dir = sample_root / "repo"
                data_dir = sample_root / "data"
                repo_dir.mkdir(parents=True)
                data_dir.mkdir(parents=True)
                materialized = materialize_sample(sample, repo_dir)
                target_path = materialized.target_path
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
                            materialized,
                        )
                    )
                rank = first_rank(ranked_paths, target_path)
                split_acc.evaluated += 1
                global_evaluated += 1
                split_acc.index_timings.append(index_ms)
                split_acc.retrieve_timings.append(retrieve_ms)
                if rank is not None:
                    split_acc.reciprocal_ranks.append(1.0 / rank)
                    for k in top_k:
                        if rank <= k:
                            split_acc.hits[k] += 1
                else:
                    split_acc.reciprocal_ranks.append(0.0)
                if args.details:
                    details_rows.append(
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

                # Bucket and level tracking
                bucket_acc.evaluated += 1
                bucket_acc.index_timings.append(index_ms)
                bucket_acc.retrieve_timings.append(retrieve_ms)
                if rank is not None:
                    bucket_acc.reciprocal_ranks.append(1.0 / rank)
                    for k in top_k:
                        if rank <= k:
                            bucket_acc.hits[k] += 1
                else:
                    bucket_acc.reciprocal_ranks.append(0.0)

                level_acc.evaluated += 1
                level_acc.index_timings.append(index_ms)
                level_acc.retrieve_timings.append(retrieve_ms)
                if rank is not None:
                    level_acc.reciprocal_ranks.append(1.0 / rank)
                    for k in top_k:
                        if rank <= k:
                            level_acc.hits[k] += 1
                else:
                    level_acc.reciprocal_ranks.append(0.0)

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                OSError,
                sqlite3.Error,
            ) as exc:
                split_acc.skipped += 1
                bucket_acc.skipped += 1
                level_acc.skipped += 1
                error_obj = {"sample_id": sample_id, "error": str(exc), "line": line_no}
                all_errors.append(error_obj)
                split_acc.errors.append(error_obj)
                bucket_acc.errors.append(error_obj)
                level_acc.errors.append(error_obj)
                if args.fail_fast:
                    raise ValueError(f"failed on {sample_id}: {exc}") from exc
            finally:
                if "sample_root" in locals() and sample_root.exists():
                    shutil.rmtree(sample_root)

    # Build split results
    split_results: dict[str, Any] = {}
    for split_name in split_names:
        acc = split_accs[split_name]
        split_results[split_name] = _build_split_result(
            split_name,
            acc,
            split_buckets[split_name],
            split_levels[split_name],
            top_k,
            dataset,
            args,
        )

    # Build combined result
    total_evaluated = sum(acc.evaluated for acc in split_accs.values())
    total_skipped = sum(acc.skipped for acc in split_accs.values())
    all_hits = {k: sum(acc.hits[k] for acc in split_accs.values()) for k in top_k}
    all_rrs = [rr for acc in split_accs.values() for rr in acc.reciprocal_ranks]
    all_index_timings = [t for acc in split_accs.values() for t in acc.index_timings]
    all_retrieve_timings = [t for acc in split_accs.values() for t in acc.retrieve_timings]

    combined_metrics = {
        f"hit_at_{k}": round(all_hits[k] / total_evaluated, 6) if total_evaluated else 0.0
        for k in top_k
    }
    combined_metrics["mrr"] = round(mean(all_rrs), 6) if all_rrs else 0.0

    return {
        "ok": True,
        "input": str(input_path),
        "dataset": dataset,
        "input_files": [f.name for f in input_files],
        "splits": split_names,
        "mode": args.mode,
        "samples_evaluated": total_evaluated,
        "samples_skipped": total_skipped,
        "top_k": top_k,
        "query_lines": args.query_lines,
        "search_limit": args.search_limit,
        "context_budget": args.context_budget,
        "start": args.start,
        "limit": args.limit,
        "metrics": combined_metrics,
        "hit_counts": {f"hit_at_{k}": all_hits[k] for k in top_k},
        "reciprocal_rank_sum": round(sum(all_rrs), 6) if all_rrs else 0.0,
        "timing_ms": {
            "index_mean": round(mean(all_index_timings), 3) if all_index_timings else 0.0,
            "retrieve_mean": round(mean(all_retrieve_timings), 3) if all_retrieve_timings else 0.0,
        },
        "split_results": split_results,
        "details": details_rows if args.details else [],
        "errors": all_errors[:20],
    }


def _build_split_result(
    split_name: str,
    acc: _Accumulator,
    buckets: dict[str, _Accumulator],
    levels: dict[str, _Accumulator],
    top_k: list[int],
    dataset: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = {
        f"hit_at_{k}": round(acc.hits[k] / acc.evaluated, 6) if acc.evaluated else 0.0
        for k in top_k
    }
    metrics["mrr"] = round(mean(acc.reciprocal_ranks), 6) if acc.reciprocal_ranks else 0.0

    by_bucket: dict[str, Any] = {}
    for bucket_name, bucket_acc in buckets.items():
        bucket_metrics = {
            f"hit_at_{k}": round(bucket_acc.hits[k] / bucket_acc.evaluated, 6)
            if bucket_acc.evaluated
            else 0.0
            for k in top_k
        }
        bucket_metrics["mrr"] = (
            round(mean(bucket_acc.reciprocal_ranks), 6) if bucket_acc.reciprocal_ranks else 0.0
        )
        by_bucket[bucket_name] = {
            "samples_evaluated": bucket_acc.evaluated,
            "samples_skipped": bucket_acc.skipped,
            "metrics": bucket_metrics,
            "hit_counts": {f"hit_at_{k}": bucket_acc.hits[k] for k in top_k},
            "reciprocal_rank_sum": round(sum(bucket_acc.reciprocal_ranks), 6)
            if bucket_acc.reciprocal_ranks
            else 0.0,
            "timing_ms": {
                "index_mean": round(mean(bucket_acc.index_timings), 3)
                if bucket_acc.index_timings
                else 0.0,
                "retrieve_mean": round(mean(bucket_acc.retrieve_timings), 3)
                if bucket_acc.retrieve_timings
                else 0.0,
            },
            "errors": bucket_acc.errors[:20],
        }

    by_level: dict[str, Any] = {}
    for level_name, level_acc in levels.items():
        level_metrics = {
            f"hit_at_{k}": round(level_acc.hits[k] / level_acc.evaluated, 6)
            if level_acc.evaluated
            else 0.0
            for k in top_k
        }
        level_metrics["mrr"] = (
            round(mean(level_acc.reciprocal_ranks), 6) if level_acc.reciprocal_ranks else 0.0
        )
        by_level[level_name] = {
            "samples_evaluated": level_acc.evaluated,
            "samples_skipped": level_acc.skipped,
            "metrics": level_metrics,
            "hit_counts": {f"hit_at_{k}": level_acc.hits[k] for k in top_k},
            "reciprocal_rank_sum": round(sum(level_acc.reciprocal_ranks), 6)
            if level_acc.reciprocal_ranks
            else 0.0,
            "timing_ms": {
                "index_mean": round(mean(level_acc.index_timings), 3)
                if level_acc.index_timings
                else 0.0,
                "retrieve_mean": round(mean(level_acc.retrieve_timings), 3)
                if level_acc.retrieve_timings
                else 0.0,
            },
            "errors": level_acc.errors[:20],
        }

    return {
        "ok": True,
        "split": split_name,
        "dataset": dataset,
        "mode": args.mode,
        "samples_evaluated": acc.evaluated,
        "samples_skipped": acc.skipped,
        "top_k": top_k,
        "query_lines": args.query_lines,
        "search_limit": args.search_limit,
        "context_budget": args.context_budget,
        "metrics": metrics,
        "hit_counts": {f"hit_at_{k}": acc.hits[k] for k in top_k},
        "reciprocal_rank_sum": round(sum(acc.reciprocal_ranks), 6) if acc.reciprocal_ranks else 0.0,
        "timing_ms": {
            "index_mean": round(mean(acc.index_timings), 3) if acc.index_timings else 0.0,
            "retrieve_mean": round(mean(acc.retrieve_timings), 3) if acc.retrieve_timings else 0.0,
        },
        "by_bucket": by_bucket,
        "by_level": by_level,
        "errors": acc.errors[:20],
    }


def _resolve_dataset(input_path: Path) -> str:
    manifest_path = input_path / ".." / "manifest.json"
    if not manifest_path.resolve().exists():
        manifest_path = input_path.parent / "manifest.json"
    if manifest_path.resolve().exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "dataset" in data:
                return str(data["dataset"])
        except (OSError, json.JSONDecodeError):
            pass
    return "tianyang/repobench_python_v1.1"


def _resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        files = [input_path]
    return files


def iter_samples(path: Path) -> Iterator[tuple[Path, int, dict[str, Any]]]:
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


def materialize_sample(sample: dict[str, Any], repo_dir: Path) -> _MaterializedSample:
    contexts = sample.get("context")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("sample has no context list")
    gold_index = int(sample["gold_snippet_index"])
    if gold_index < 0 or gold_index >= len(contexts):
        raise IndexError("gold_snippet_index is outside context list")

    used_paths: set[str] = set()
    target_path = ""
    candidates: list[_MaterializedCandidate] = []
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
        candidates.append(
            _MaterializedCandidate(
                path=rel.as_posix(),
                identifier=str(context.get("identifier") or ""),
                snippet=snippet,
                identifier_terms=frozenset(
                    _identifier_tokens(str(context.get("identifier") or ""))
                ),
                path_terms=frozenset(_path_terms(rel.as_posix())),
                snippet_terms=frozenset(_identifier_tokens(snippet[:4096])),
                order=index,
            )
        )
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
    return _MaterializedSample(
        target_path=target_path,
        current_path=current_path.as_posix(),
        cropped_code=cropped_code,
        candidates=candidates,
    )


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
    selected = lines[-max(1, query_lines) :]
    identifiers: list[str] = []
    for line in reversed(selected):
        identifiers.extend(_identifier_tokens(line, preserve_case=True))
    identifiers.extend(_import_query_terms(cropped_code, preserve_case=True))
    identifiers = _dedupe_preserve_order(identifiers)
    return " ".join(identifiers[:96]) if identifiers else "\n".join(selected)


def retrieve_paths(
    conn: sqlite3.Connection,
    query: str,
    mode: str,
    search_limit: int,
    context_budget: int,
    materialized: _MaterializedSample | None = None,
) -> list[str]:
    paths: list[str] = []
    if mode in {"search", "hybrid"}:
        paths.extend(row["path"] for row in search_nodes(conn, query, limit=search_limit))
    if mode in {"context", "hybrid"}:
        pack = build_context_pack(
            conn,
            query,
            budget=context_budget,
            limit=min(search_limit, 20),
            include_related=False,
        )
        paths.extend(item["path"] for item in pack.get("top_hits") or [])
        paths.extend(item["path"] for item in pack.get("must_read") or [])
    paths = unique_preserve_order(paths)
    if materialized is not None:
        return rerank_materialized_candidates(query, materialized, paths, search_limit)
    return paths


def rerank_materialized_candidates(
    query: str,
    materialized: _MaterializedSample,
    lode_paths: list[str],
    limit: int,
) -> list[str]:
    query_terms = _weighted_query_terms(query)
    import_terms = set(_import_query_terms(materialized.cropped_code))
    lode_rank = {path: index for index, path in enumerate(lode_paths)}
    ranked: list[tuple[float, int, int, str]] = []
    for candidate in materialized.candidates:
        score = _candidate_score(candidate, query_terms, import_terms)
        if candidate.path in lode_rank:
            score += max(0.0, 12.0 - float(lode_rank[candidate.path])) * 0.75
        if score <= 0.0:
            continue
        ranked.append(
            (
                -score,
                lode_rank.get(candidate.path, len(lode_paths) + candidate.order),
                candidate.order,
                candidate.path,
            )
        )
    candidate_paths = [path for *_unused, path in sorted(ranked)]
    return unique_preserve_order([*candidate_paths, *lode_paths])[:limit]


def _candidate_score(
    candidate: _MaterializedCandidate,
    query_terms: dict[str, float],
    import_terms: set[str],
) -> float:
    if not query_terms and not import_terms:
        return 0.0
    identifier_terms = candidate.identifier_terms
    path_terms = candidate.path_terms
    snippet_terms = candidate.snippet_terms
    query_set = set(query_terms)

    score = 0.0
    score += 8.0 * len(query_set & identifier_terms)
    score += 6.0 * len(query_set & path_terms)
    score += 1.0 * len(query_set & snippet_terms)
    score += 12.0 * len(import_terms & path_terms)
    score += 10.0 * len(import_terms & identifier_terms)
    score += 0.75 * len(import_terms & snippet_terms)

    for term, weight in query_terms.items():
        if term in identifier_terms:
            score += 4.0 * weight
        if term in path_terms:
            score += 3.0 * weight
        if term in snippet_terms:
            score += 0.35 * weight
    return score


def _weighted_query_terms(query: str) -> dict[str, float]:
    terms = _identifier_tokens(query)
    weights: dict[str, float] = {}
    for index, term in enumerate(terms[:32]):
        weights[term] = max(weights.get(term, 0.0), 1.0 - min(index, 31) / 40.0)
    return weights


def _import_query_terms(code: str, preserve_case: bool = False) -> list[str]:
    terms: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        terms.extend(_identifier_tokens(stripped, preserve_case=preserve_case))
        terms.extend(_path_terms(stripped, preserve_case=preserve_case))
    return [term for term in _dedupe_preserve_order(terms) if term.lower() not in _STOPWORDS]


def _identifier_tokens(value: str, preserve_case: bool = False) -> list[str]:
    tokens: list[str] = []
    for raw in _IDENTIFIER_RE.findall(value):
        tokens.extend(_split_identifier(raw, preserve_case=preserve_case))
    return [token for token in tokens if token.lower() not in _STOPWORDS]


def _path_terms(value: str, preserve_case: bool = False) -> list[str]:
    tokens: list[str] = []
    for raw in _PATH_TOKEN_RE.findall(value.replace("\\", "/")):
        tokens.extend(_split_identifier(raw, preserve_case=preserve_case))
    return [token for token in tokens if token.lower() not in _STOPWORDS]


def _split_identifier(value: str, preserve_case: bool = False) -> list[str]:
    parts = [
        part
        for chunk in re.split(r"[_\W]+", value)
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", chunk)
    ]
    if not parts and value:
        parts = [value]
    tokens = [part if preserve_case else part.lower() for part in parts if len(part) > 1]
    if len(value) > 1:
        tokens.append(value if preserve_case else value.lower())
    return _dedupe_preserve_order(tokens)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


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
    print(f"dataset: {result.get('dataset', 'unknown')}")
    print(f"mode: {result['mode']}")
    print(f"splits: {', '.join(result.get('splits', []))}")
    print(f"samples: evaluated={result['samples_evaluated']} skipped={result['samples_skipped']}")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.6f}")
    print(
        "timing mean ms: "
        f"index={result['timing_ms']['index_mean']:.3f} "
        f"retrieve={result['timing_ms']['retrieve_mean']:.3f}"
    )
    if result.get("split_results"):
        print("split results:")
        for split_name, split_data in result["split_results"].items():
            print(
                f"  {split_name}: evaluated={split_data['samples_evaluated']} skipped={split_data['samples_skipped']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
