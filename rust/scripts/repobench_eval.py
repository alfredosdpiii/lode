"""RepoBench-R retrieval eval for the Rust `lode` binary.

Same sampling, buckets and metrics as /tmp/lode_hillclimb_eval.py, with retrieval
done by `lode index` + `lode context` subprocesses. `--python` adds the original
in-process Python path for side-by-side numbers.

Usage (from the repo root):
    uv run python rust/scripts/repobench_eval.py [per_bucket=50] [query_lines=5] [--rust-bin PATH] [--python]
        [--merge-paths [--rerank]]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.repobench_adapter import (  # noqa: E402
    first_rank,
    iter_samples,
    materialize_sample,
    query_from_sample,
    safe_relative_path,
    unique_preserve_order,
)
from lode.config import sqlite_path  # noqa: E402
from lode.context import build_context_pack  # noqa: E402
from lode.indexer import index_repo  # noqa: E402
from lode.storage import connect  # noqa: E402

BENCH_DIR = REPO / "bench-results/20260531T203339Z-full-repobench-r"
BUCKETS = ("lt5", "easy", "hard")
BUDGET = 6000
LIMIT = 30
SUBPROCESS_TIMEOUT_S = 600
RERANK = False


class RustError(RuntimeError):
    pass


def find_rust_bin(explicit: str | None) -> Path:
    candidates = (
        [Path(explicit)]
        if explicit
        else [REPO / "rust/target/release/lode", REPO / "rust/target/debug/lode"]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    print(
        f"error: Rust lode binary not found (looked at: {', '.join(map(str, candidates))})",
        file=sys.stderr,
    )
    sys.exit(2)


def run_rust(rust_bin: Path, data_dir: Path, args: list[str]) -> tuple[dict[str, Any], float]:
    """Run the Rust CLI; return (parsed JSON, wall ms). Any failure raises RustError."""
    cmd = [str(rust_bin), "--data-dir", str(data_dir), *args]
    t0 = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S, check=False
        )  # nosec B603
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RustError(f"{args[0]}: {exc}") from exc
    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000
    if proc.returncode != 0:
        raise RustError(f"{args[0]} exit {proc.returncode}: {proc.stderr.strip()[-400:]}")
    try:
        return json.loads(proc.stdout), wall_ms
    except json.JSONDecodeError as exc:
        raise RustError(f"{args[0]}: bad JSON ({exc}): {proc.stdout[:200]!r}") from exc


def rust_index(rust_bin: Path, repo_dir: Path, data_dir: Path) -> tuple[dict[str, Any], float]:
    return run_rust(rust_bin, data_dir, ["index", str(repo_dir), "--json"])


def rust_context(
    rust_bin: Path,
    data_dir: Path,
    query: str,
    *,
    budget: int = BUDGET,
    limit: int = LIMIT,
    from_path: str | None = None,
    exclude: set[str] | None = None,
    include_related: bool = True,
    rerank: bool = False,
) -> tuple[dict[str, Any], float]:
    args = ["context", query, "--json", "--budget", str(budget), "--limit", str(limit)]
    if rerank:
        args.append("--rerank")
    if from_path:
        args += ["--from-path", from_path]
    for path in sorted(exclude or ()):
        args += ["--exclude-path", path]
    if not include_related:
        args.append("--no-related")
    return run_rust(rust_bin, data_dir, args)


class BatchContext:
    """One `lode eval-batch` process for all queries, so the rerank model loads once."""

    def __init__(self, rust_bin: Path) -> None:
        env = {**os.environ, "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "8")}
        self.proc = subprocess.Popen(  # nosec B603
            [str(rust_bin), "eval-batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )

    def context(
        self,
        data_dir: Path,
        query: str,
        rerank_query: str,
        *,
        from_path: str,
        exclude: set[str],
        limit: int = LIMIT,
        **_: Any,
    ) -> tuple[dict[str, Any], float]:
        assert self.proc.stdin and self.proc.stdout
        request = {
            "data_dir": str(data_dir),
            "query": query,
            "rerank_query": rerank_query,
            "from_path": from_path,
            "exclude": sorted(exclude),
            "limit": limit,
            "rerank": True,
        }
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(
            self.proc.stdout.readline() or '{"ok": false, "error": "eval-batch exited"}'
        )
        if not reply["ok"]:
            raise RustError(f"eval-batch: {reply['error']}")
        return reply["pack"], reply["ms"]

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=60)


def rerank_query_of(sample: dict[str, Any], lines: int = 20) -> str:
    code = [line for line in str(sample.get("cropped_code") or "").splitlines() if line.strip()]
    return "\n".join(code[-lines:])


def python_context(
    data_dir: Path,
    query: str,
    *,
    budget: int = BUDGET,
    limit: int = LIMIT,
    from_path: str | None = None,
    exclude: set[str] | None = None,
    include_related: bool = True,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter_ns()
    with closing(connect(sqlite_path(data_dir))) as conn:
        pack = build_context_pack(
            conn,
            query,
            budget=budget,
            limit=limit,
            include_related=include_related,
            from_path=from_path,
            exclude_paths=exclude,
        )
    return pack, (time.perf_counter_ns() - t0) / 1_000_000


def pack_paths(pack: dict[str, Any]) -> list[str]:
    return unique_preserve_order(
        [item["path"] for item in pack.get("top_hits") or []]
        + [item["path"] for item in pack.get("must_read") or []]
    )


def bucket_of(sample: dict[str, Any]) -> str:
    ctx = sample.get("context") or []
    return "lt5" if len(ctx) < 5 else ("easy" if len(ctx) <= 9 else "hard")


def select_samples(jsonl_path: Path, per_bucket: int) -> list[tuple[int, dict[str, Any]]]:
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {b: [] for b in BUCKETS}
    for _, line_no, sample in iter_samples(jsonl_path):
        bucket = buckets[bucket_of(sample)]
        if len(bucket) < per_bucket:
            bucket.append((line_no, sample))
        if all(len(v) >= per_bucket for v in buckets.values()):
            break
    return [item for v in buckets.values() for item in v]


def prepare_sample(sample: dict[str, Any], repo_dir: Path) -> Any:
    """Materialize a sample and prepend its import statement to the current file."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    mat = materialize_sample(sample, repo_dir)
    imp = str(sample.get("import_statement") or "").strip()
    if imp:
        cur_file = repo_dir / mat.current_path
        cur_file.write_text(imp + "\n\n" + cur_file.read_text(encoding="utf-8"), encoding="utf-8")
    return mat


def empty_stats() -> dict[str, dict[str, float]]:
    keys = ("n", "h1", "h3", "h5", "h10", "mrr", "zero", "missing", "errors", "ret_ms")
    return {b: dict.fromkeys(keys, 0) for b in BUCKETS}


def record(st: dict[str, float], paths: list[str], target: str, ret_ms: float) -> None:
    st["n"] += 1
    st["ret_ms"] += ret_ms
    rank = first_rank(paths, target)
    if rank is not None:
        st["mrr"] += 1.0 / rank
        for k in (1, 3, 5, 10):
            if rank <= k:
                st[f"h{k}"] += 1
    elif not paths:
        st["zero"] += 1
    else:
        st["missing"] += 1


def print_stats(title: str, stats: dict[str, dict[str, float]]) -> None:
    print(title)
    for b, st in stats.items():
        n = max(1, st["n"])
        print(
            f"  {b:4s} (n={int(st['n']):3d}, err={int(st['errors'])}): "
            f"H@1={st['h1'] / n:.3f} H@3={st['h3'] / n:.3f} H@5={st['h5'] / n:.3f} H@10={st['h10'] / n:.3f} "
            f"MRR={st['mrr'] / n:.3f} | zero={int(st['zero'])} miss={int(st['missing'])} ret={st['ret_ms'] / n:.2f}ms"
        )


def eval_split(
    jsonl_path: Path, per_bucket: int, query_lines: int, rust_bin: Path | None, with_python: bool
) -> dict[str, dict[str, dict[str, float]]]:
    sides = (["rust"] if rust_bin else []) + (["python"] if with_python else [])
    stats = {side: empty_stats() for side in sides}
    with tempfile.TemporaryDirectory(prefix="lode-rs-eval-") as tmp:
        for idx, (_, sample) in enumerate(select_samples(jsonl_path, per_bucket)):
            b = bucket_of(sample)
            sdir = Path(tmp) / f"s{idx}"
            try:
                mat = prepare_sample(sample, sdir / "repo")
            except Exception:
                for side in sides:
                    stats[side][b]["errors"] += 1
                continue
            q = query_from_sample(sample, query_lines)
            ctx_kw: dict[str, Any] = {
                "from_path": mat.current_path,
                "exclude": {mat.current_path},
                "include_related": False,
            }
            if rust_bin:
                data = sdir / "rs-data"
                data.mkdir()
                try:
                    rust_index(rust_bin, sdir / "repo", data)
                    pack, ms = rust_context(rust_bin, data, q, **ctx_kw)
                    record(stats["rust"][b], pack_paths(pack), mat.target_path, ms)
                except RustError as exc:
                    stats["rust"][b]["errors"] += 1
                    print(f"  [rust error] {jsonl_path.stem} s{idx}: {exc}", file=sys.stderr)
            if with_python:
                data = sdir / "py-data"
                data.mkdir()
                try:
                    index_repo(sdir / "repo", sqlite_path(data))
                    pack, ms = python_context(data, q, **ctx_kw)
                    record(stats["python"][b], pack_paths(pack), mat.target_path, ms)
                except Exception as exc:
                    stats["python"][b]["errors"] += 1
                    print(f"  [python error] {jsonl_path.stem} s{idx}: {exc}", file=sys.stderr)

    for side in sides:
        note = " [ret includes process startup]" if side == "rust" else " [ret in-process]"
        print_stats(
            f"=== {jsonl_path.stem} [{side}] (per_bucket={per_bucket}, query_lines={query_lines}){note} ===",
            stats[side],
        )
    return stats


def prepare_merged(
    sample: dict[str, Any], repo_dir: Path
) -> tuple[str, list[tuple[str, int, int]], int]:
    """Write snippets sharing a path into one file, like the real repository.

    Returns (current_path, [(path, start_line, end_line) per snippet], gold index).
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    by_path: dict[str, list[int]] = {}
    contexts = sample["context"]
    for index, ctx in enumerate(contexts):
        rel = safe_relative_path(ctx.get("path"), f"context_{index}.py").as_posix()
        by_path.setdefault(rel, []).append(index)
    spans: list[tuple[str, int, int]] = [("", 0, 0)] * len(contexts)
    for rel, indexes in by_path.items():
        lines: list[str] = []
        for index in indexes:
            snippet = str(contexts[index].get("snippet") or "").rstrip().splitlines() or [""]
            start = len(lines) + 1
            lines.extend(snippet)
            spans[index] = (rel, start, len(lines))
            lines.append("")
        (repo_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo_dir / rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
    current = safe_relative_path(sample.get("file_path"), "current.py").as_posix()
    if current in by_path:
        current = current[:-3] + "__current.py"
    imp = str(sample.get("import_statement") or "").strip()
    code = str(sample.get("cropped_code") or "")
    (repo_dir / current).parent.mkdir(parents=True, exist_ok=True)
    (repo_dir / current).write_text(
        (imp + "\n\n" if imp else "") + code.rstrip() + "\n", encoding="utf-8"
    )
    return current, spans, int(sample["gold_snippet_index"])


def snippet_ranking(pack: dict[str, Any], spans: list[tuple[str, int, int]]) -> list[int]:
    """Map ranked hits to snippet indexes: a node hit picks the snippets its lines overlap,
    a File hit picks every snippet of that file in file order."""
    ranked: list[int] = []
    for hit in pack.get("top_hits") or []:
        start, end = hit["lines"]
        for index, (path, s, e) in enumerate(spans):
            if path != hit["path"] or index in ranked:
                continue
            if hit["kind"] == "File" or (start <= e and s <= end):
                ranked.append(index)
    return ranked


def eval_split_merged(
    jsonl_path: Path, per_bucket: int, query_lines: int, rust_bin: Path | None, with_python: bool
) -> dict[str, dict[str, dict[str, float]]]:
    sides = (["rust"] if rust_bin else []) + (["rust_rerank"] if rust_bin and RERANK else [])
    sides += ["python"] if with_python else []
    stats = {side: empty_stats() for side in sides}
    batch = BatchContext(rust_bin) if rust_bin and RERANK else None
    with tempfile.TemporaryDirectory(prefix="lode-rs-merge-") as tmp:
        for idx, (_, sample) in enumerate(select_samples(jsonl_path, per_bucket)):
            b = bucket_of(sample)
            sdir = Path(tmp) / f"s{idx}"
            try:
                current, spans, gold = prepare_merged(sample, sdir / "repo")
            except Exception:
                for side in sides:
                    stats[side][b]["errors"] += 1
                continue
            q = query_from_sample(sample, query_lines)
            ctx_kw: dict[str, Any] = {
                "from_path": current,
                "exclude": {current},
                "include_related": False,
            }
            for side in sides:
                data = sdir / f"{side}-data"
                data.mkdir()
                try:
                    if side.startswith("rust"):
                        assert rust_bin is not None
                        rust_index(rust_bin, sdir / "repo", data)
                        if side == "rust_rerank":
                            assert batch is not None
                            pack, ms = batch.context(data, q, rerank_query_of(sample), **ctx_kw)
                        else:
                            pack, ms = rust_context(rust_bin, data, q, **ctx_kw)
                    else:
                        index_repo(sdir / "repo", sqlite_path(data))
                        pack, ms = python_context(data, q, **ctx_kw)
                    ranked = [str(i) for i in snippet_ranking(pack, spans)]
                    record(stats[side][b], ranked, str(gold), ms)
                except Exception as exc:
                    stats[side][b]["errors"] += 1
                    print(f"  [{side} error] {jsonl_path.stem} s{idx}: {exc}", file=sys.stderr)
    if batch:
        batch.close()
    for side in sides:
        print_stats(
            f"=== {jsonl_path.stem} [{side}] MERGED snippet-level (per_bucket={per_bucket}) ===",
            stats[side],
        )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("per_bucket", nargs="?", type=int, default=50)
    parser.add_argument("query_lines", nargs="?", type=int, default=5)
    parser.add_argument("--rust-bin", default=None)
    parser.add_argument("--python", action="store_true", help="also run the in-process Python path")
    parser.add_argument(
        "--python-only", action="store_true", help="skip Rust (sanity check of the harness)"
    )
    parser.add_argument(
        "--merge-paths",
        action="store_true",
        help="write snippets sharing a path into one file and score snippets, not files",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="with --merge-paths, also score `lode context --rerank` (binary built with --features embed)",
    )
    args = parser.parse_args()
    global RERANK
    RERANK = args.rerank
    rust_bin = None if args.python_only else find_rust_bin(args.rust_bin)
    with_python = args.python or args.python_only
    for split in ("cross_file_first", "cross_file_random"):
        (eval_split_merged if args.merge_paths else eval_split)(
            BENCH_DIR / f"{split}.jsonl", args.per_bucket, args.query_lines, rust_bin, with_python
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
