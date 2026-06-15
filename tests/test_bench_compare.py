from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_COMPARE = [sys.executable, "scripts/bench_compare.py"]


class BenchCompareOperationalTests(unittest.TestCase):
    def test_comparator_passes_better_metrics(self) -> None:
        current = load_baseline()
        # Improve every latency metric so it beats baseline
        current["cold_index"]["timing_ms"] = 100.0
        current["hot_index"]["timing_ms"] = 5.0
        # Fix timing summaries to remain valid (min <= p50 <= p95 <= max)
        for path in [
            "search.build context pack.timing_ms",
            "search.embedding queue.timing_ms",
            "symbols.build_context_pack.timing_ms",
            "context.build context pack.timing_ms",
            "context.embedding queue.timing_ms",
            "neighbors.timing_ms",
        ]:
            summary = get_nested(current, path)
            if summary:
                summary["p50"] = 0.1
                summary["min"] = 0.05
                summary["p95"] = 0.15
                summary["max"] = 0.2
        current["kuzu"]["timing_ms"] = 1000.0
        current["embeddings"]["vectors_per_second"] = 50.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [
                    *BENCH_COMPARE,
                    "--type",
                    "operational",
                    "--current",
                    str(current_path),
                    "--command",
                    "uv run python scripts/bench_lode.py --repo . --json",
                    "--env",
                    "LODE_EMBEDDINGS_MODEL=Snowflake/snowflake-arctic-embed-s",
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["overall_pass"], msg=json.dumps(payload, indent=2))
            self.assertEqual(payload["type"], "operational")
            self.assertIn("baseline_source", payload)
            self.assertIn("metrics", payload)
            self.assertEqual(
                payload["command"], "uv run python scripts/bench_lode.py --repo . --json"
            )
            self.assertEqual(
                payload["environment"]["LODE_EMBEDDINGS_MODEL"],
                "Snowflake/snowflake-arctic-embed-s",
            )
            for path, metric in payload["metrics"].items():
                self.assertIn("baseline", metric)
                self.assertIn("current", metric)
                self.assertIn("direction", metric)
                self.assertIn("delta", metric)
                self.assertIn("pass", metric)
            self.assertNotIn("missing_metrics", payload)
            self.assertNotIn("failed_metrics", payload)
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_worsened_metric(self) -> None:
        current = load_baseline()
        # Make one latency metric worse than baseline
        current["cold_index"]["timing_ms"] = 500.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("cold_index.timing_ms", payload["failed_metrics"])
            self.assertEqual(payload["metrics"]["cold_index.timing_ms"]["pass"], False)
            self.assertEqual(
                payload["metrics"]["cold_index.timing_ms"]["direction"], "lower_is_better"
            )
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_missing_metric(self) -> None:
        current = load_baseline()
        del current["hot_index"]["timing_ms"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("missing_metrics", payload)
            self.assertIn("hot_index.timing_ms", payload["missing_metrics"])
            self.assertEqual(payload["metrics"]["hot_index.timing_ms"]["pass"], False)
            self.assertEqual(payload["metrics"]["hot_index.timing_ms"]["reason"], "missing metric")
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_invariant_violation(self) -> None:
        current = load_baseline()
        current["embeddings"]["dims"] = 512

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("embeddings.dims", payload["failed_metrics"])
            self.assertEqual(payload["metrics"]["embeddings.dims"]["direction"], "invariant")
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_coverage_floor_violation(self) -> None:
        current = load_baseline()
        current["database"]["counts"]["nodes"] = 100

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("database.counts.nodes", payload["failed_metrics"])
            self.assertEqual(
                payload["metrics"]["database.counts.nodes"]["direction"], "invariant_floor"
            )
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_reports_timing_issues(self) -> None:
        current = load_baseline()
        # Corrupt timing summary order
        current["search"]["build context pack"]["timing_ms"]["p50"] = 10.0
        current["search"]["build context pack"]["timing_ms"]["min"] = 20.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("timing_issues", payload)
            issues = [i for i in payload["timing_issues"] if "order violation" in i]
            self.assertTrue(len(issues) > 0)
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_nonnumeric_metric(self) -> None:
        current = load_baseline()
        current["neighbors"]["timing_ms"]["p50"] = "fast"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("missing_metrics", payload)
            self.assertIn("neighbors.timing_ms.p50", payload["missing_metrics"])
            self.assertEqual(
                payload["metrics"]["neighbors.timing_ms.p50"]["reason"], "non-numeric metric"
            )
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_output_file(self) -> None:
        current = load_baseline()
        current["cold_index"]["timing_ms"] = 100.0
        current["hot_index"]["timing_ms"] = 5.0
        current["kuzu"]["timing_ms"] = 1000.0
        current["embeddings"]["vectors_per_second"] = 50.0
        # Fix timing summaries to remain valid
        for path in [
            "search.build context pack.timing_ms",
            "search.embedding queue.timing_ms",
            "symbols.build_context_pack.timing_ms",
            "context.build context pack.timing_ms",
            "context.embedding queue.timing_ms",
            "neighbors.timing_ms",
        ]:
            summary = get_nested(current, path)
            if summary:
                summary["p50"] = 0.1
                summary["min"] = 0.05
                summary["p95"] = 0.15
                summary["max"] = 0.2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        output_path = Path(tempfile.mktemp(suffix=".json"))
        try:
            result = subprocess.run(
                [
                    *BENCH_COMPARE,
                    "--type",
                    "operational",
                    "--current",
                    str(current_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertTrue(output_path.exists())
            file_payload = json.loads(output_path.read_text())
            stdout_payload = json.loads(result.stdout)
            self.assertEqual(file_payload, stdout_payload)
        finally:
            current_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class BenchCompareRepoBenchTests(unittest.TestCase):
    def test_repobench_comparator_passes_better_metrics(self) -> None:
        current = load_repobench_baseline()
        # Improve all quality metrics and lower latency
        for key in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"]:
            current["metrics"][key] = 0.99
        current["timing_ms"]["retrieve_mean"] = 0.01
        for split in ["cross_file_first", "cross_file_random"]:
            for key in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"]:
                current["split_results"][split]["metrics"][key] = 0.99
            current["split_results"][split]["timing_ms"]["retrieve_mean"] = 0.01

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "repobench", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["overall_pass"], msg=json.dumps(payload, indent=2))
            self.assertEqual(payload["type"], "repobench")
            self.assertIn("baseline_source", payload)
            self.assertIn("metrics", payload)
            self.assertIn("split_results", payload)
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_worsened_metric(self) -> None:
        current = load_repobench_baseline()
        current["metrics"]["hit_at_1"] = 0.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "repobench", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("metrics.hit_at_1", payload["failed_metrics"])
            self.assertEqual(
                payload["metrics"]["metrics.hit_at_1"]["direction"], "higher_is_better"
            )
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_missing_mrr(self) -> None:
        current = load_repobench_baseline()
        del current["metrics"]["mrr"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "repobench", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("missing_metrics", payload)
            self.assertIn("metrics.mrr", payload["missing_metrics"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_invariant_sample_count(self) -> None:
        current = load_repobench_baseline()
        current["samples_evaluated"] = 15000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "repobench", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("samples_evaluated", payload["failed_metrics"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_missing_split(self) -> None:
        current = load_repobench_baseline()
        del current["split_results"]["cross_file_first"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "repobench", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("missing_metrics", payload)
            self.assertIn("split_results.cross_file_first", payload["missing_metrics"])
        finally:
            current_path.unlink(missing_ok=True)


class BenchCompareSmokeTests(unittest.TestCase):
    def test_missing_current_file_errors(self) -> None:
        result = subprocess.run(
            [*BENCH_COMPARE, "--type", "operational", "--current", "/tmp/does_not_exist.json"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stderr)

    def test_invalid_type_errors(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump({}, fh)
            current_path = Path(fh.name)
        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "invalid", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid choice", result.stderr)
        finally:
            current_path.unlink(missing_ok=True)


def load_baseline() -> dict:
    path = PROJECT_ROOT / "bench-results" / "20260531T184011Z" / "lode.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def load_repobench_baseline() -> dict:
    path = (
        PROJECT_ROOT
        / "bench-results"
        / "20260531T203339Z-full-repobench-r"
        / "lode-context-ql5-combined.json"
    )
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def get_nested(obj: dict, dotted: str) -> dict | None:
    parts = dotted.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return cast(Any, None)
        node = node.get(part)
        if node is None:
            return cast(Any, None)
    return node if isinstance(node, dict) else None


if __name__ == "__main__":
    unittest.main()
