#!/usr/bin/env python3
"""Benchmark comparator harness for Lode operational, embedding, and RepoBench workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPERATIONAL_BASELINE = PROJECT_ROOT / "bench-results" / "20260531T184011Z" / "lode.json"
REPOBENCH_BASELINE = (
    PROJECT_ROOT
    / "bench-results"
    / "20260531T203339Z-full-repobench-r"
    / "lode-context-ql5-combined.json"
)

APPROVED_PARAMETERS: dict[str, dict[str, Any]] = {
    "operational": {
        "repo": "/home/bryan/Projects/lode",
        "repeat": 10,
        "limit": 20,
        "budget": 4000,
        "queries": ["build context pack", "embedding queue"],
        "symbols": ["build_context_pack"],
        "include_kuzu": True,
    },
    "embedding": {
        "repo": "/home/bryan/Projects/lode",
        "repeat": 10,
        "limit": 20,
        "budget": 4000,
        "queries": ["build context pack", "embedding queue"],
        "symbols": ["build_context_pack"],
        "embed_url": "http://127.0.0.1:7980",
        "embed_limit": 32,
        "model": "Snowflake/snowflake-arctic-embed-s",
    },
    "repobench": {
        "input": "/home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl",
        "mode": "context",
        "top_k": [1, 3, 5, 10],
        "query_lines": 5,
        "search_limit": 30,
        "context_budget": 6000,
        "limit": None,
        "start": 0,
    },
}

OPERATIONAL_METRICS: list[dict[str, Any]] = [
    {"path": "cold_index.timing_ms", "direction": "lower_is_better"},
    {"path": "hot_index.timing_ms", "direction": "lower_is_better"},
    {"path": "search.build context pack.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "search.embedding queue.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "symbols.build_context_pack.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "context.build context pack.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "context.embedding queue.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "neighbors.timing_ms.p50", "direction": "lower_is_better"},
    {"path": "kuzu.timing_ms", "direction": "lower_is_better"},
    {"path": "database.counts.files", "direction": "invariant_floor", "expected": 19},
    {"path": "database.counts.nodes", "direction": "invariant_floor", "expected": 814},
    {"path": "database.counts.edges", "direction": "invariant_floor", "expected": 1166},
]

OPERATIONAL_TIMING_PATHS = [
    "search.build context pack.timing_ms",
    "search.embedding queue.timing_ms",
    "symbols.build_context_pack.timing_ms",
    "context.build context pack.timing_ms",
    "context.embedding queue.timing_ms",
    "neighbors.timing_ms",
]

EMBEDDING_METRICS: list[dict[str, Any]] = [
    {"path": "embeddings.vectors_per_second", "direction": "higher_is_better"},
    {"path": "embeddings.dims", "direction": "invariant", "expected": 384},
    {"path": "embeddings.embedded", "direction": "invariant", "expected": 32},
    {
        "path": "embeddings.model",
        "direction": "invariant",
        "expected": "Snowflake/snowflake-arctic-embed-s",
    },
]

EMBEDDING_TIMING_PATHS = [
    "search.build context pack.timing_ms",
    "search.embedding queue.timing_ms",
    "symbols.build_context_pack.timing_ms",
    "context.build context pack.timing_ms",
    "context.embedding queue.timing_ms",
    "neighbors.timing_ms",
]

REPOBENCH_METRICS: list[dict[str, Any]] = [
    {"path": "metrics.hit_at_1", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_3", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_5", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_10", "direction": "higher_is_better"},
    {"path": "metrics.mrr", "direction": "higher_is_better"},
    {"path": "timing_ms.retrieve_mean", "direction": "lower_is_better"},
    {"path": "samples_evaluated", "direction": "invariant", "expected": 15636},
    {"path": "samples_skipped", "direction": "invariant", "expected": 15},
]

REPOBENCH_SPLIT_METRICS: list[dict[str, Any]] = [
    {"path": "metrics.hit_at_1", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_3", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_5", "direction": "higher_is_better"},
    {"path": "metrics.hit_at_10", "direction": "higher_is_better"},
    {"path": "metrics.mrr", "direction": "higher_is_better"},
    {"path": "timing_ms.retrieve_mean", "direction": "lower_is_better"},
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_path = args.baseline
    current_path = args.current
    command = args.command
    env = parse_env(args.env)

    if not current_path.exists():
        emit_error(f"Current results file does not exist: {current_path}")
        return 1

    with current_path.open("r", encoding="utf-8") as fh:
        current = json.load(fh)

    if args.type == "operational":
        if not baseline_path:
            baseline_path = OPERATIONAL_BASELINE
        result = compare_operational(baseline_path, current, command, env)
    elif args.type == "embedding":
        if not baseline_path:
            baseline_path = OPERATIONAL_BASELINE
        result = compare_embedding(baseline_path, current, command, env)
    elif args.type == "repobench":
        if not baseline_path:
            baseline_path = REPOBENCH_BASELINE
        result = compare_repobench(baseline_path, current, command, env)
    else:
        emit_error(f"Unknown comparison type: {args.type}")
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("overall_pass") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results against historical baselines."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["operational", "embedding", "repobench"],
        help="Benchmark type",
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Current benchmark results JSON"
    )
    parser.add_argument("--baseline", type=Path, help="Override baseline JSON path")
    parser.add_argument("--command", help="Exact command used to generate current results")
    parser.add_argument("--env", action="append", default=[], help="Environment variable key=value")
    parser.add_argument("--output", type=Path, help="Write comparison result to this file")
    return parser


def parse_env(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            env[key] = value
    return env


def emit_error(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)


def extract_parameters(current: dict[str, Any], comp_type: str) -> dict[str, Any]:
    """Extract normalized parameters from current benchmark JSON."""
    if comp_type in {"operational", "embedding"}:
        params = current.get("parameters", {})
        if not params:
            params = {
                "repo": current.get("repo"),
            }
        return params
    if comp_type == "repobench":
        return {
            "input": current.get("input"),
            "mode": current.get("mode"),
            "top_k": current.get("top_k"),
            "query_lines": current.get("query_lines"),
            "search_limit": current.get("search_limit"),
            "context_budget": current.get("context_budget"),
            "limit": current.get("limit"),
            "start": current.get("start"),
        }
    return {}


def validate_parameters(
    current: dict[str, Any], approved: dict[str, Any], _comp_type: str
) -> list[dict[str, Any]]:
    """Compare current parameters against approved parameters and return diagnostics."""
    diagnostics: list[dict[str, Any]] = []

    for key, expected in approved.items():
        actual = current.get(key)
        if key == "repo":
            expected_path = Path(expected).resolve()
            actual_path = Path(actual).resolve() if actual else None
            if actual_path != expected_path:
                diagnostics.append(
                    {
                        "parameter": key,
                        "expected": str(expected_path),
                        "actual": str(actual_path) if actual_path else None,
                        "message": f"repo path mismatch: expected {expected_path}, got {actual_path}",
                    }
                )
        elif key in {"queries", "symbols", "top_k"}:
            if isinstance(expected, list) and isinstance(actual, list):
                if expected != actual:
                    diagnostics.append(
                        {
                            "parameter": key,
                            "expected": expected,
                            "actual": actual,
                            "message": f"{key} mismatch: expected {expected}, got {actual}",
                        }
                    )
            elif actual != expected:
                diagnostics.append(
                    {
                        "parameter": key,
                        "expected": expected,
                        "actual": actual,
                        "message": f"{key} mismatch: expected {expected}, got {actual}",
                    }
                )
        elif key == "limit":
            if expected is None and actual is not None:
                diagnostics.append(
                    {
                        "parameter": key,
                        "expected": expected,
                        "actual": actual,
                        "message": f"repobench limit must be absent (full run), got {actual}",
                    }
                )
            elif expected is not None and actual != expected:
                diagnostics.append(
                    {
                        "parameter": key,
                        "expected": expected,
                        "actual": actual,
                        "message": f"{key} mismatch: expected {expected}, got {actual}",
                    }
                )
        elif actual != expected:
            diagnostics.append(
                {
                    "parameter": key,
                    "expected": expected,
                    "actual": actual,
                    "message": f"{key} mismatch: expected {expected}, got {actual}",
                }
            )

    return diagnostics


def _compare_metrics(
    baseline: dict[str, Any],
    current: dict[str, Any],
    metrics_spec: list[dict[str, Any]],
    prefix: str = "",
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Compare metrics against baseline and return (metrics_out, missing, failed)."""
    metrics_out: dict[str, Any] = {}
    missing: list[str] = []
    failed: list[str] = []

    for spec in metrics_spec:
        path = spec["path"]
        full_path = f"{prefix}{path}" if prefix else path
        baseline_val = get_path(baseline, path)
        current_val = get_path(current, path)

        if baseline_val is None or current_val is None:
            missing.append(full_path)
            metrics_out[path] = {
                "baseline": baseline_val,
                "current": current_val,
                "direction": spec["direction"],
                "delta": None,
                "pass": False,  # nosec B105
                "reason": "missing metric",
            }
            continue

        if spec["direction"] != "invariant" or isinstance(spec.get("expected"), (int, float)):
            if not isinstance(baseline_val, (int, float)) or not isinstance(
                current_val, (int, float)
            ):
                missing.append(full_path)
                metrics_out[path] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "direction": spec["direction"],
                    "delta": None,
                    "pass": False,  # nosec B105
                    "reason": "non-numeric metric",
                }
                continue
            delta = round(current_val - baseline_val, 6)
        else:
            delta = None
        direction = spec["direction"]
        if direction == "lower_is_better":
            passed = current_val < baseline_val
        elif direction == "higher_is_better":
            passed = current_val > baseline_val
        elif direction == "invariant":
            passed = current_val == spec["expected"]
        elif direction == "invariant_floor":
            passed = current_val >= spec["expected"]
        else:
            passed = False

        if not passed:
            failed.append(full_path)

        metrics_out[path] = {
            "baseline": baseline_val,
            "current": current_val,
            "direction": direction,
            "delta": delta,
            "pass": passed,
        }

    return metrics_out, missing, failed


def _check_timing_summaries(
    current: dict[str, Any], timing_paths: list[str], expected_count: int = 10
) -> list[str]:
    """Check timing summary completeness and ordering."""
    timing_issues: list[str] = []
    for timing_path in timing_paths:
        summary = get_path(current, timing_path)
        if summary is None:
            continue
        required_keys = ["count", "min", "p50", "mean", "p95", "max"]
        for key in required_keys:
            if key not in summary:
                timing_issues.append(f"{timing_path}.{key}")
            else:
                val = summary[key]
                if not isinstance(val, (int, float)):
                    timing_issues.append(f"{timing_path}.{key}: non-numeric")
        if summary and all(k in summary for k in required_keys):
            count = summary["count"]
            if count != expected_count:
                timing_issues.append(f"{timing_path}.count != {expected_count}")
            vals = [summary["min"], summary["p50"], summary["p95"], summary["max"]]
            if all(isinstance(v, (int, float)) for v in vals):
                if not (0 <= vals[0] <= vals[1] <= vals[2] <= vals[3]):
                    timing_issues.append(f"{timing_path}: order violation min<=p50<=p95<=max")
    return timing_issues


def compare_operational(
    baseline_path: Path, current: dict[str, Any], command: str | None, env: dict[str, str]
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    metrics_out, missing, failed = _compare_metrics(baseline, current, OPERATIONAL_METRICS)
    overall_pass = not missing and not failed

    timing_issues = _check_timing_summaries(current, OPERATIONAL_TIMING_PATHS)
    if timing_issues:
        overall_pass = False

    current_params = extract_parameters(current, "operational")
    approved_params = APPROVED_PARAMETERS["operational"]
    param_diagnostics = validate_parameters(current_params, approved_params, "operational")
    if param_diagnostics:
        overall_pass = False

    result: dict[str, Any] = {
        "ok": True,
        "baseline_source": str(baseline_path),
        "type": "operational",
        "command": command,
        "environment": env,
        "approved_parameters": approved_params,
        "current_parameters": current_params,
        "overall_pass": overall_pass,
        "metrics": metrics_out,
    }
    if missing:
        result["missing_metrics"] = missing
    if failed:
        result["failed_metrics"] = failed
    if timing_issues:
        result["timing_issues"] = timing_issues
    if param_diagnostics:
        result["parameter_diagnostics"] = param_diagnostics
    return result


def compare_embedding(
    baseline_path: Path, current: dict[str, Any], command: str | None, env: dict[str, str]
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    metrics_out, missing, failed = _compare_metrics(baseline, current, EMBEDDING_METRICS)
    overall_pass = not missing and not failed

    timing_issues = _check_timing_summaries(current, EMBEDDING_TIMING_PATHS)
    if timing_issues:
        overall_pass = False

    # Database reconciliation invariant
    reconciliation_issues: list[str] = []
    before = get_path(current, "database.counts.embeddings")
    after = get_path(current, "database_after_embeddings.counts.embeddings")
    embedded = get_path(current, "embeddings.embedded")
    if before is not None and after is not None and embedded is not None:
        if after != before + embedded:
            reconciliation_issues.append(
                f"database_after_embeddings.counts.embeddings ({after}) != "
                f"database.counts.embeddings ({before}) + embeddings.embedded ({embedded})"
            )
            overall_pass = False
    elif any(v is not None for v in (before, after, embedded)):
        reconciliation_issues.append("incomplete embedding database reconciliation data")
        overall_pass = False

    current_params = extract_parameters(current, "embedding")
    approved_params = APPROVED_PARAMETERS["embedding"]
    param_diagnostics = validate_parameters(current_params, approved_params, "embedding")
    if param_diagnostics:
        overall_pass = False

    result: dict[str, Any] = {
        "ok": True,
        "baseline_source": str(baseline_path),
        "type": "embedding",
        "command": command,
        "environment": env,
        "approved_parameters": approved_params,
        "current_parameters": current_params,
        "overall_pass": overall_pass,
        "metrics": metrics_out,
    }
    if missing:
        result["missing_metrics"] = missing
    if failed:
        result["failed_metrics"] = failed
    if timing_issues:
        result["timing_issues"] = timing_issues
    if reconciliation_issues:
        result["reconciliation_issues"] = reconciliation_issues
    if param_diagnostics:
        result["parameter_diagnostics"] = param_diagnostics
    return result


def compare_repobench(
    baseline_path: Path, current: dict[str, Any], command: str | None, env: dict[str, str]
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    metrics_out, missing, failed = _compare_metrics(baseline, current, REPOBENCH_METRICS)
    overall_pass = not missing and not failed

    # Combined evaluated/skipped invariant
    invariant_issues: list[str] = []
    baseline_total = get_path(baseline, "samples_evaluated")
    baseline_skipped = get_path(baseline, "samples_skipped")
    current_total = get_path(current, "samples_evaluated")
    current_skipped = get_path(current, "samples_skipped")
    if (
        baseline_total is not None
        and baseline_skipped is not None
        and current_total is not None
        and current_skipped is not None
    ):
        expected_combined = baseline_total + baseline_skipped
        actual_combined = current_total + current_skipped
        if actual_combined != expected_combined:
            invariant_issues.append(
                f"combined evaluated + skipped mismatch: expected {expected_combined}, got {actual_combined}"
            )
            overall_pass = False

    # Split checks
    split_results: dict[str, Any] = {}
    for split in ["cross_file_first", "cross_file_random"]:
        baseline_split: dict[str, Any] | None = get_path(baseline, f"split_results.{split}")
        current_split: dict[str, Any] | None = get_path(current, f"split_results.{split}")
        if baseline_split is None or current_split is None:
            if current_split is None:
                missing.append(f"split_results.{split}")
                overall_pass = False
            continue

        split_metrics, split_missing, split_failed = _compare_metrics(
            baseline_split, current_split, REPOBENCH_SPLIT_METRICS, prefix=f"split_results.{split}."
        )
        missing.extend(split_missing)
        failed.extend(split_failed)
        if split_missing or split_failed:
            overall_pass = False

        # Per-split evaluated/skipped invariant
        baseline_split_eval = get_path(baseline_split, "samples_evaluated")
        baseline_split_skip = get_path(baseline_split, "samples_skipped")
        current_split_eval = get_path(current_split, "samples_evaluated")
        current_split_skip = get_path(current_split, "samples_skipped")
        if (
            baseline_split_eval is not None
            and baseline_split_skip is not None
            and current_split_eval is not None
            and current_split_skip is not None
        ):
            expected_split_total = baseline_split_eval + baseline_split_skip
            actual_split_total = current_split_eval + current_split_skip
            if actual_split_total != expected_split_total:
                invariant_issues.append(
                    f"split {split} evaluated + skipped mismatch: expected {expected_split_total}, got {actual_split_total}"
                )
                overall_pass = False

        split_results[split] = split_metrics

    current_params = extract_parameters(current, "repobench")
    approved_params = APPROVED_PARAMETERS["repobench"]
    param_diagnostics = validate_parameters(current_params, approved_params, "repobench")
    if param_diagnostics:
        overall_pass = False

    result: dict[str, Any] = {
        "ok": True,
        "baseline_source": str(baseline_path),
        "type": "repobench",
        "command": command,
        "environment": env,
        "approved_parameters": approved_params,
        "current_parameters": current_params,
        "overall_pass": overall_pass,
        "metrics": metrics_out,
        "split_results": split_results,
    }
    if missing:
        result["missing_metrics"] = missing
    if failed:
        result["failed_metrics"] = failed
    if invariant_issues:
        result["invariant_issues"] = invariant_issues
    if param_diagnostics:
        result["parameter_diagnostics"] = param_diagnostics
    return result


def get_path(obj: dict[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return cast(Any, None)
        node = node.get(part)
        if node is None:
            return cast(Any, None)
    return node


if __name__ == "__main__":
    raise SystemExit(main())
