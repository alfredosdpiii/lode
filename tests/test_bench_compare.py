from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "benchmarks" / "baselines"

BENCH_COMPARE = [sys.executable, "scripts/bench_compare.py"]

OPERATIONAL_PARAMETERS = {
    "repo": "/home/bryan/Projects/lode",
    "repeat": 10,
    "limit": 20,
    "budget": 4000,
    "queries": ["build context pack", "embedding queue"],
    "symbols": ["build_context_pack"],
    "include_kuzu": True,
    "embed_url": None,
    "embed_limit": 32,
}

EMBEDDING_PARAMETERS = {
    "repo": "/home/bryan/Projects/lode",
    "repeat": 10,
    "limit": 20,
    "budget": 4000,
    "queries": ["build context pack", "embedding queue"],
    "symbols": ["build_context_pack"],
    "embed_url": "http://127.0.0.1:7980",
    "embed_limit": 32,
    "model": "Snowflake/snowflake-arctic-embed-s",
}

REPOBENCH_PARAMETERS = {
    "input": "/home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl",
    "mode": "context",
    "context_include_related": True,
    "top_k": [1, 3, 5, 10],
    "query_lines": 5,
    "search_limit": 30,
    "context_budget": 6000,
    "limit": None,
    "start": 0,
}


class BenchCompareOperationalTests(unittest.TestCase):
    def test_comparator_passes_better_metrics(self) -> None:
        current = load_baseline()
        current["parameters"] = OPERATIONAL_PARAMETERS
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
            self.assertIn("approved_parameters", payload)
            self.assertIn("current_parameters", payload)
            for path, metric in payload["metrics"].items():
                self.assertIn("baseline", metric)
                self.assertIn("current", metric)
                self.assertIn("direction", metric)
                self.assertIn("delta", metric)
                self.assertIn("pass", metric)
            self.assertNotIn("missing_metrics", payload)
            self.assertNotIn("failed_metrics", payload)
            self.assertNotIn("parameter_diagnostics", payload)
            # Operational must not require embeddings metrics
            self.assertNotIn("embeddings.vectors_per_second", payload["metrics"])
            self.assertNotIn("embeddings.dims", payload["metrics"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_comparator_fails_worsened_metric(self) -> None:
        current = load_baseline()
        current["parameters"] = OPERATIONAL_PARAMETERS
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
        current["parameters"] = OPERATIONAL_PARAMETERS
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

    def test_comparator_fails_coverage_floor_violation(self) -> None:
        current = load_baseline()
        current["parameters"] = OPERATIONAL_PARAMETERS
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
        current["parameters"] = OPERATIONAL_PARAMETERS
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
        current["parameters"] = OPERATIONAL_PARAMETERS
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
        current["parameters"] = OPERATIONAL_PARAMETERS
        current["cold_index"]["timing_ms"] = 100.0
        current["hot_index"]["timing_ms"] = 5.0
        current["kuzu"]["timing_ms"] = 1000.0
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

        output_fd, output_name = tempfile.mkstemp(suffix=".json")
        os.close(output_fd)
        output_path = Path(output_name)
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

    def test_operational_ignores_bad_embedding_metrics(self) -> None:
        current = load_baseline()
        current["parameters"] = OPERATIONAL_PARAMETERS
        # Make embedding metrics terrible: operational should still ignore them
        current["embeddings"]["vectors_per_second"] = 0.1
        current["embeddings"]["dims"] = 512
        current["embeddings"]["embedded"] = 1
        current["embeddings"]["model"] = "wrong-model"
        # Improve operational metrics so they pass independently
        current["cold_index"]["timing_ms"] = 100.0
        current["hot_index"]["timing_ms"] = 5.0
        current["kuzu"]["timing_ms"] = 1000.0
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

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "operational", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            # Should pass because operational does not require embeddings
            self.assertTrue(payload["overall_pass"], msg=json.dumps(payload, indent=2))
            self.assertNotIn("embeddings.vectors_per_second", payload["metrics"])
            self.assertNotIn("embeddings.dims", payload["metrics"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_operational_reports_parameter_mismatch(self) -> None:
        current = load_baseline()
        current["parameters"] = {**OPERATIONAL_PARAMETERS, "repeat": 5}

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
            self.assertIn("parameter_diagnostics", payload)
            diagnostics = payload["parameter_diagnostics"]
            repeat_diag = [d for d in diagnostics if d["parameter"] == "repeat"]
            self.assertEqual(len(repeat_diag), 1)
            self.assertEqual(repeat_diag[0]["expected"], 10)
            self.assertEqual(repeat_diag[0]["actual"], 5)
        finally:
            current_path.unlink(missing_ok=True)

    def test_operational_fails_search_top_path_regression(self) -> None:
        current = make_passing_operational_current()
        current["search"]["embedding queue"]["top_path"] = "README.md"

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
            self.assertIn("search.embedding queue.top_path", payload["failed_metrics"])
            self.assertEqual(
                payload["metrics"]["search.embedding queue.top_path"]["current"],
                "README.md",
            )
        finally:
            current_path.unlink(missing_ok=True)

    def test_operational_fails_context_confidence_regression(self) -> None:
        current = make_passing_operational_current()
        current["context"]["build context pack"]["confidence"] = "exact"

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
            self.assertIn("context.build context pack.confidence", payload["failed_metrics"])
            self.assertEqual(
                payload["metrics"]["context.build context pack.confidence"]["current"],
                "exact",
            )
        finally:
            current_path.unlink(missing_ok=True)


class BenchCompareEmbeddingTests(unittest.TestCase):
    def test_embedding_comparator_passes_better_metrics(self) -> None:
        current = load_baseline()
        current["parameters"] = EMBEDDING_PARAMETERS
        current["embeddings"]["vectors_per_second"] = 50.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["overall_pass"], msg=json.dumps(payload, indent=2))
            self.assertEqual(payload["type"], "embedding")
            self.assertIn("baseline_source", payload)
            self.assertIn("metrics", payload)
            self.assertIn("embeddings.vectors_per_second", payload["metrics"])
            self.assertIn("embeddings.dims", payload["metrics"])
            self.assertIn("approved_parameters", payload)
            self.assertIn("current_parameters", payload)
            self.assertNotIn("parameter_diagnostics", payload)
        finally:
            current_path.unlink(missing_ok=True)

    def test_embedding_comparator_fails_worsened_throughput(self) -> None:
        current = load_baseline()
        current["parameters"] = EMBEDDING_PARAMETERS
        current["embeddings"]["vectors_per_second"] = 1.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("failed_metrics", payload)
            self.assertIn("embeddings.vectors_per_second", payload["failed_metrics"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_embedding_comparator_fails_invariant_violation(self) -> None:
        current = load_baseline()
        current["parameters"] = EMBEDDING_PARAMETERS
        current["embeddings"]["dims"] = 512

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
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

    def test_embedding_comparator_fails_model_mismatch(self) -> None:
        current = load_baseline()
        current["parameters"] = {**EMBEDDING_PARAMETERS, "model": "wrong-model"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("parameter_diagnostics", payload)
            model_diag = [d for d in payload["parameter_diagnostics"] if d["parameter"] == "model"]
            self.assertEqual(len(model_diag), 1)
            self.assertEqual(model_diag[0]["expected"], "Snowflake/snowflake-arctic-embed-s")
            self.assertEqual(model_diag[0]["actual"], "wrong-model")
        finally:
            current_path.unlink(missing_ok=True)

    def test_embedding_comparator_fails_database_reconciliation(self) -> None:
        current = load_baseline()
        current["parameters"] = EMBEDDING_PARAMETERS
        # Break the reconciliation: make after != before + embedded
        current["database_after_embeddings"]["counts"]["embeddings"] = 9999

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("reconciliation_issues", payload)
            self.assertTrue(len(payload["reconciliation_issues"]) > 0)
        finally:
            current_path.unlink(missing_ok=True)

    def test_embedding_comparator_fails_embed_limit_mismatch(self) -> None:
        current = load_baseline()
        current["parameters"] = {**EMBEDDING_PARAMETERS, "embed_limit": 64}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(current, fh)
            current_path = Path(fh.name)

        try:
            result = subprocess.run(
                [*BENCH_COMPARE, "--type", "embedding", "--current", str(current_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["overall_pass"])
            self.assertIn("parameter_diagnostics", payload)
            limit_diag = [
                d for d in payload["parameter_diagnostics"] if d["parameter"] == "embed_limit"
            ]
            self.assertEqual(len(limit_diag), 1)
            self.assertEqual(limit_diag[0]["expected"], 32)
            self.assertEqual(limit_diag[0]["actual"], 64)
        finally:
            current_path.unlink(missing_ok=True)


class BenchCompareRepoBenchTests(unittest.TestCase):
    def test_repobench_comparator_passes_better_metrics(self) -> None:
        current = load_repobench_baseline()
        # Inject parameters so validation passes
        current["input"] = "/home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl"
        current["start"] = 0
        current["limit"] = None
        current["context_include_related"] = True
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
            self.assertIn("approved_parameters", payload)
            self.assertIn("current_parameters", payload)
            self.assertTrue(payload["current_parameters"]["context_include_related"])
            self.assertNotIn("parameter_diagnostics", payload)
            self.assertNotIn("invariant_issues", payload)
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_missing_context_include_related(self) -> None:
        current = load_repobench_baseline()
        current["input"] = "/home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl"
        current["start"] = 0
        current["limit"] = None
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
            self.assertFalse(payload["overall_pass"])
            diagnostics = payload["parameter_diagnostics"]
            context_diag = [d for d in diagnostics if d["parameter"] == "context_include_related"]
            self.assertEqual(len(context_diag), 1)
            self.assertTrue(context_diag[0]["expected"])
            self.assertIsNone(context_diag[0]["actual"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_disabled_context_include_related(self) -> None:
        current = load_repobench_baseline()
        current["input"] = "/home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl"
        current["start"] = 0
        current["limit"] = None
        current["context_include_related"] = False
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
            self.assertFalse(payload["overall_pass"])
            diagnostics = payload["parameter_diagnostics"]
            context_diag = [d for d in diagnostics if d["parameter"] == "context_include_related"]
            self.assertEqual(len(context_diag), 1)
            self.assertTrue(context_diag[0]["expected"])
            self.assertFalse(context_diag[0]["actual"])
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_worsened_metric(self) -> None:
        current = load_repobench_baseline()
        current["start"] = 0
        current["limit"] = None
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
        current["start"] = 0
        current["limit"] = None
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
        current["start"] = 0
        current["limit"] = None
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
        current["start"] = 0
        current["limit"] = None
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

    def test_repobench_comparator_fails_combined_evaluated_skipped_invariant(self) -> None:
        current = load_repobench_baseline()
        current["start"] = 0
        current["limit"] = None
        # Break the invariant: evaluated + skipped != baseline total
        current["samples_evaluated"] = 15600
        current["samples_skipped"] = 10

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
            self.assertIn("invariant_issues", payload)
            issues = [i for i in payload["invariant_issues"] if "combined evaluated + skipped" in i]
            self.assertEqual(len(issues), 1)
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_split_evaluated_skipped_invariant(self) -> None:
        current = load_repobench_baseline()
        current["start"] = 0
        current["limit"] = None
        # Break split invariant for cross_file_first
        current["split_results"]["cross_file_first"]["samples_evaluated"] = 8000
        current["split_results"]["cross_file_first"]["samples_skipped"] = 5

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
            self.assertIn("invariant_issues", payload)
            issues = [i for i in payload["invariant_issues"] if "cross_file_first" in i]
            self.assertEqual(len(issues), 1)
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_split_swapped_evaluated_skipped(self) -> None:
        """Swap evaluated and skipped while keeping total constant; exact counts must fail."""
        current = load_repobench_baseline()
        current["start"] = 0
        current["limit"] = None
        # Swap evaluated and skipped for cross_file_first: total stays 8033
        current["split_results"]["cross_file_first"]["samples_evaluated"] = 7
        current["split_results"]["cross_file_first"]["samples_skipped"] = 8026

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
            # Should fail with split and field diagnostics
            self.assertIn("failed_metrics", payload)
            self.assertIn(
                "split_results.cross_file_first.samples_evaluated", payload["failed_metrics"]
            )
            self.assertIn(
                "split_results.cross_file_first.samples_skipped", payload["failed_metrics"]
            )
            # Verify split_results contains the diagnostic details
            self.assertIn("split_results", payload)
            split_diag = payload["split_results"]["cross_file_first"]
            self.assertIn("samples_evaluated", split_diag)
            self.assertEqual(split_diag["samples_evaluated"]["pass"], False)
            self.assertEqual(split_diag["samples_evaluated"]["baseline"], 8026)
            self.assertEqual(split_diag["samples_evaluated"]["current"], 7)
            self.assertIn("samples_skipped", split_diag)
            self.assertEqual(split_diag["samples_skipped"]["pass"], False)
            self.assertEqual(split_diag["samples_skipped"]["baseline"], 7)
            self.assertEqual(split_diag["samples_skipped"]["current"], 8026)
        finally:
            current_path.unlink(missing_ok=True)

    def test_repobench_comparator_fails_parameter_mismatch(self) -> None:
        current = load_repobench_baseline()
        current["start"] = 0
        current["limit"] = None
        current["mode"] = "search"

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
            self.assertIn("parameter_diagnostics", payload)
            mode_diag = [d for d in payload["parameter_diagnostics"] if d["parameter"] == "mode"]
            self.assertEqual(len(mode_diag), 1)
            self.assertEqual(mode_diag[0]["expected"], "context")
            self.assertEqual(mode_diag[0]["actual"], "search")
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
    path = BASELINE_DIR / "lode-operational-20260531.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def make_passing_operational_current() -> dict:
    current = load_baseline()
    current["parameters"] = OPERATIONAL_PARAMETERS
    current["cold_index"]["timing_ms"] = 100.0
    current["hot_index"]["timing_ms"] = 5.0
    current["kuzu"]["timing_ms"] = 1000.0
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
    return current


def load_repobench_baseline() -> dict:
    path = BASELINE_DIR / "repobench-python-v1.1-context-20260531.json"
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
