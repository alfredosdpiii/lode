#!/usr/bin/env python3
"""Benchmark comparator harness for Lode operational and RepoBench workloads."""

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
    {"path": "embeddings.vectors_per_second", "direction": "higher_is_better"},
    {"path": "embeddings.dims", "direction": "invariant", "expected": 384},
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
        "--type", required=True, choices=["operational", "repobench"], help="Benchmark type"
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


def compare_operational(
    baseline_path: Path, current: dict[str, Any], command: str | None, env: dict[str, str]
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    metrics_out: dict[str, Any] = {}
    overall_pass = True
    missing: list[str] = []
    failed: list[str] = []

    for spec in OPERATIONAL_METRICS:
        path = spec["path"]
        baseline_val = get_path(baseline, path)
        current_val = get_path(current, path)

        if baseline_val is None or current_val is None:
            missing.append(path)
            overall_pass = False
            metrics_out[path] = {
                "baseline": baseline_val,
                "current": current_val,
                "direction": spec["direction"],
                "delta": None,
                "pass": False,  # nosec B105
                "reason": "missing metric",
            }
            continue

        if not isinstance(baseline_val, (int, float)) or not isinstance(current_val, (int, float)):
            missing.append(path)
            overall_pass = False
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
            failed.append(path)
            overall_pass = False

        metrics_out[path] = {
            "baseline": baseline_val,
            "current": current_val,
            "direction": direction,
            "delta": delta,
            "pass": passed,
        }

    # Timing summary completeness check
    timing_issues: list[str] = []
    for timing_path in OPERATIONAL_TIMING_PATHS:
        summary = get_path(current, timing_path)
        if summary is None:
            continue
        required_keys = ["count", "min", "p50", "mean", "p95", "max"]
        for key in required_keys:
            if key not in summary:
                timing_issues.append(f"{timing_path}.{key}")
                overall_pass = False
            else:
                val = summary[key]
                if not isinstance(val, (int, float)):
                    timing_issues.append(f"{timing_path}.{key}: non-numeric")
                    overall_pass = False
        if summary and all(k in summary for k in required_keys):
            count = summary["count"]
            if count != 10:
                timing_issues.append(f"{timing_path}.count != 10")
                overall_pass = False
            vals = [summary["min"], summary["p50"], summary["p95"], summary["max"]]
            if all(isinstance(v, (int, float)) for v in vals):
                if not (0 <= vals[0] <= vals[1] <= vals[2] <= vals[3]):
                    timing_issues.append(f"{timing_path}: order violation min<=p50<=p95<=max")
                    overall_pass = False

    result: dict[str, Any] = {
        "ok": True,
        "baseline_source": str(baseline_path),
        "type": "operational",
        "command": command,
        "environment": env,
        "overall_pass": overall_pass,
        "metrics": metrics_out,
    }
    if missing:
        result["missing_metrics"] = missing
    if failed:
        result["failed_metrics"] = failed
    if timing_issues:
        result["timing_issues"] = timing_issues
    return result


def compare_repobench(
    baseline_path: Path, current: dict[str, Any], command: str | None, env: dict[str, str]
) -> dict[str, Any]:
    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    metrics_out: dict[str, Any] = {}
    overall_pass = True
    missing: list[str] = []
    failed: list[str] = []

    for spec in REPOBENCH_METRICS:
        path = spec["path"]
        baseline_val = get_path(baseline, path)
        current_val = get_path(current, path)

        if baseline_val is None or current_val is None:
            missing.append(path)
            overall_pass = False
            metrics_out[path] = {
                "baseline": baseline_val,
                "current": current_val,
                "direction": spec["direction"],
                "delta": None,
                "pass": False,  # nosec B105
                "reason": "missing metric",
            }
            continue

        if not isinstance(baseline_val, (int, float)) or not isinstance(current_val, (int, float)):
            missing.append(path)
            overall_pass = False
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
        direction = spec["direction"]
        if direction == "lower_is_better":
            passed = current_val < baseline_val
        elif direction == "higher_is_better":
            passed = current_val > baseline_val
        elif direction == "invariant":
            passed = current_val == spec["expected"]
        else:
            passed = False

        if not passed:
            failed.append(path)
            overall_pass = False

        metrics_out[path] = {
            "baseline": baseline_val,
            "current": current_val,
            "direction": direction,
            "delta": delta,
            "pass": passed,
        }

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

        split_metrics: dict[str, Any] = {}
        for spec in REPOBENCH_SPLIT_METRICS:
            path = spec["path"]
            baseline_val = get_path(baseline_split, path)
            current_val = get_path(current_split, path)
            if baseline_val is None or current_val is None:
                missing.append(f"split_results.{split}.{path}")
                overall_pass = False
                split_metrics[path] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "direction": spec["direction"],
                    "delta": None,
                    "pass": False,  # nosec B105
                    "reason": "missing metric",
                }
                continue

            if not isinstance(baseline_val, (int, float)) or not isinstance(
                current_val, (int, float)
            ):
                missing.append(f"split_results.{split}.{path}")
                overall_pass = False
                split_metrics[path] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "direction": spec["direction"],
                    "delta": None,
                    "pass": False,  # nosec B105
                    "reason": "non-numeric metric",
                }
                continue

            delta = round(current_val - baseline_val, 6)
            direction = spec["direction"]
            if direction == "lower_is_better":
                passed = current_val < baseline_val
            elif direction == "higher_is_better":
                passed = current_val > baseline_val
            else:
                passed = False

            if not passed:
                failed.append(f"split_results.{split}.{path}")
                overall_pass = False

            split_metrics[path] = {
                "baseline": baseline_val,
                "current": current_val,
                "direction": direction,
                "delta": delta,
                "pass": passed,
            }
        split_results[split] = split_metrics

    result: dict[str, Any] = {
        "ok": True,
        "baseline_source": str(baseline_path),
        "type": "repobench",
        "command": command,
        "environment": env,
        "overall_pass": overall_pass,
        "metrics": metrics_out,
        "split_results": split_results,
    }
    if missing:
        result["missing_metrics"] = missing
    if failed:
        result["failed_metrics"] = failed
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
