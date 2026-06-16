from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_gate_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lode_final_quality_gate", PROJECT_ROOT / "scripts" / "final_quality_gate.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load final quality gate script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_contract(path: Path, assertion_ids: list[str]) -> None:
    lines = ["# Contract", ""]
    for assertion_id in assertion_ids:
        lines.extend([f"### {assertion_id}: test assertion", "", "Tool: shell", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_state(path: Path, assertion_ids: list[str], *, pending: set[str] | None = None) -> None:
    pending = pending or set()
    assertions: dict[str, dict[str, Any]] = {}
    for assertion_id in assertion_ids:
        assertions[assertion_id] = (
            {"status": "pending"}
            if assertion_id in pending
            else {
                "status": "passed",
                "validatedAtMilestone": "fixture",
                "artifactPaths": ["/tmp/prior-evidence.json"],
            }
        )
    path.write_text(json.dumps({"assertions": assertions}), encoding="utf-8")


class FakeCommandExecutor:
    def __init__(
        self,
        gate_module: Any,
        *,
        events: list[str] | None = None,
        pollute_repo: Path | None = None,
        hosted_benchmark_output: bool = False,
        fail_compare: str | None = None,
        fail_compare_attempts: dict[str, int] | None = None,
        fail_command_names: set[str] | None = None,
    ) -> None:
        self._gate_module = gate_module
        self._events = events
        self._pollute_repo = pollute_repo
        self._hosted_benchmark_output = hosted_benchmark_output
        self._fail_compare = fail_compare
        self._fail_compare_attempts = fail_compare_attempts or {}
        self._fail_command_names = fail_command_names or set()
        self.calls: list[str] = []
        self.names: list[str] = []
        self._compare_prefix_counts: dict[str, int] = {}

    def run(
        self,
        spec: Any,
        *,
        cwd: Path,
        env: dict[str, str],
        output_dir: Path,
        timeout: int | None,
    ) -> Any:
        _ = (cwd, env, timeout)
        self.calls.append(spec.command)
        self.names.append(spec.name)
        if self._events is not None:
            self._events.append(f"command:{spec.name}")
        if self._pollute_repo is not None and spec.name == "quality_format_check":
            (self._pollute_repo / "pollution.txt").write_text("generated", encoding="utf-8")

        stdout = ""
        if spec.name.startswith("benchmark_"):
            stdout = json.dumps({"ok": True, "parameters": {}})
            if self._hosted_benchmark_output and spec.name.startswith("benchmark_operational"):
                stdout = json.dumps({"ok": True, "log": "unexpected https://example.com/upload"})
        if spec.name.startswith("compare_"):
            exit_code = (
                1
                if self._fail_compare is not None and spec.name.startswith(self._fail_compare)
                else 0
            )
            for prefix, fail_count in self._fail_compare_attempts.items():
                if spec.name.startswith(prefix):
                    current_count = self._compare_prefix_counts.get(prefix, 0) + 1
                    self._compare_prefix_counts[prefix] = current_count
                    if current_count <= fail_count:
                        exit_code = 1
            stdout = json.dumps(
                {
                    "ok": True,
                    "overall_pass": exit_code == 0,
                    "metrics": {"fixture.metric": {"pass": exit_code == 0}},
                    "split_results": {},
                }
            )
        else:
            exit_code = 1 if spec.name in self._fail_command_names else 0
        stdout_path = output_dir / f"{spec.name}.stdout"
        stderr_path = output_dir / f"{spec.name}.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return self._gate_module.CommandResult(
            name=spec.name,
            command=spec.command,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            duration_seconds=0.01,
            stdout_preview=stdout,
            stderr_preview="",
        )


class FakeServiceManager:
    def __init__(
        self, gate_module: Any, *, fail_stop: bool = False, events: list[str] | None = None
    ) -> None:
        self._gate_module = gate_module
        self._fail_stop = fail_stop
        self._events = events
        self.started: list[str] = []
        self.stopped: list[str] = []

    def ensure_started(self, name: str, evidence_dir: Path) -> Any:
        _ = evidence_dir
        self.started.append(name)
        if self._events is not None:
            self._events.append(f"service_start:{name}")
        return self._gate_module.ServiceRun(name=name, started_by_gate=True, ok=True, evidence=[])

    def stop_started(self, run: Any, evidence_dir: Path) -> Any:
        _ = evidence_dir
        self.stopped.append(run.name)
        if self._events is not None:
            self._events.append(f"service_stop:{run.name}")
        return self._gate_module.ServiceStop(
            name=run.name,
            ok=not self._fail_stop,
            evidence=[],
            failure_reason="fixture stop failure" if self._fail_stop else "",
        )


class FinalQualityGateTests(unittest.TestCase):
    def test_validation_run_id_parser_reuses_existing_validation_networks(self) -> None:
        gate_module = load_gate_script()

        run_ids = gate_module.parse_validation_run_ids(
            [
                "bridge",
                "lode-validation-gate-2142295_default",
                "lode-validation-fa6cf798e9b3_default",
                "lode_default",
            ]
        )

        self.assertEqual(run_ids, ["fa6cf798e9b3", "gate-2142295"])

    def test_gate_couples_quality_benchmarks_comparisons_and_matrix(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = [
            "VAL-BENCH-001",
            "VAL-OPER-005",
            "VAL-REPOBENCH-007",
            "VAL-CROSS-001",
            "VAL-CROSS-004",
            "VAL-CROSS-005",
            "VAL-CROSS-006",
            "VAL-CROSS-007",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)

            executor = FakeCommandExecutor(gate_module)
            service_manager = FakeServiceManager(gate_module)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=evidence_dir,
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=executor,
                service_manager=service_manager,
            )

            result = gate.run()

        self.assertTrue(result["ok"], msg=json.dumps(result, indent=2))
        quality_calls = [
            command for command in executor.calls if command in gate_module.QUALITY_COMMANDS
        ]
        self.assertEqual(quality_calls, gate_module.QUALITY_COMMANDS)
        self.assertLess(
            executor.names.index("benchmark_operational_attempt_1"),
            executor.names.index("quality_format_check"),
        )
        for command in (
            gate_module.approved_operational_command(PROJECT_ROOT),
            gate_module.approved_embedding_command(PROJECT_ROOT),
            gate_module.approved_repobench_command(PROJECT_ROOT),
        ):
            self.assertIn(command, executor.calls)
        self.assertIn("embeddings", service_manager.started)
        self.assertIn("embeddings", service_manager.stopped)
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(set(matrix), set(assertion_ids))
        self.assertTrue(all(row["status"] == "pass" for row in matrix.values()))
        self.assertEqual(result["checks"]["quality_gate"]["status"], "pass")
        self.assertEqual(result["checks"]["benchmark_comparison"]["status"], "pass")
        self.assertIn("LODE_EMBEDDINGS_URL", result["local_first"]["cleared_environment_keys"])
        for path in result["artifacts"]:
            self.assertFalse(Path(path).resolve().is_relative_to(PROJECT_ROOT.resolve()))

    def test_gate_fails_when_assertion_state_has_pending_row(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-BENCH-001", "VAL-CROSS-001", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids, pending={"VAL-BENCH-001"})
            executor = FakeCommandExecutor(gate_module)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=executor,
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-BENCH-001"]["status"], "fail")
        self.assertIn("pending", matrix["VAL-BENCH-001"]["failure_reason"])
        self.assertEqual(matrix["VAL-CROSS-007"]["status"], "fail")

    def test_gate_detects_generated_repo_pollution(self) -> None:
        gate_module = load_gate_script()
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as gate_tmp:
            repo = Path(repo_tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            contract_path = Path(gate_tmp) / "validation-contract.md"
            state_path = Path(gate_tmp) / "validation-state.json"
            assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-005", "VAL-CROSS-007"]
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)

            executor = FakeCommandExecutor(gate_module, pollute_repo=repo)
            gate = gate_module.FinalQualityGate(
                project_root=repo,
                evidence_dir=Path(gate_tmp) / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=executor,
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["checks"]["artifact_hygiene"]["status"], "fail")
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-CROSS-005"]["status"], "fail")
        self.assertIn("git status changed", matrix["VAL-CROSS-005"]["failure_reason"])

    def test_gate_fails_when_service_cleanup_fails(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-004", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            service_manager = FakeServiceManager(gate_module, fail_stop=True)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=FakeCommandExecutor(gate_module),
                service_manager=service_manager,
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["checks"]["cleanup"]["status"], "fail")
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-CROSS-004"]["status"], "fail")
        self.assertIn("fixture stop failure", matrix["VAL-CROSS-004"]["failure_reason"])
        self.assertEqual(matrix["VAL-CROSS-007"]["status"], "fail")

    def test_gate_fails_when_local_first_scan_finds_hosted_endpoint(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-006", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=FakeCommandExecutor(gate_module, hosted_benchmark_output=True),
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["local_first"]["status"], "fail")
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-CROSS-006"]["status"], "fail")
        self.assertIn("hosted URL detected", matrix["VAL-CROSS-006"]["failure_reason"])
        self.assertEqual(matrix["VAL-CROSS-007"]["status"], "fail")

    def test_gate_fails_when_benchmark_comparison_fails(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=FakeCommandExecutor(
                    gate_module, fail_compare="compare_operational"
                ),
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["checks"]["benchmark_comparison"]["status"], "fail")
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-CROSS-001"]["status"], "fail")
        self.assertIn("benchmark", matrix["VAL-CROSS-001"]["failure_reason"])
        self.assertEqual(matrix["VAL-CROSS-007"]["status"], "fail")

    def test_gate_runs_repobench_before_embedding_service_and_warms_tei(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-004", "VAL-CROSS-007"]
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            executor = FakeCommandExecutor(gate_module, events=events)
            service_manager = FakeServiceManager(gate_module, events=events)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=executor,
                service_manager=service_manager,
            )

            result = gate.run()

        self.assertTrue(result["ok"], msg=json.dumps(result, indent=2))
        operational = events.index("command:benchmark_operational_attempt_1")
        repobench = events.index("command:benchmark_repobench_attempt_1")
        service_start = events.index("service_start:embeddings")
        warmup = events.index("command:benchmark_embedding_warmup")
        embedding = events.index("command:benchmark_embeddings_attempt_1")
        service_stop = events.index("service_stop:embeddings")
        self.assertLess(operational, repobench)
        self.assertLess(repobench, service_start)
        self.assertLess(service_start, warmup)
        self.assertLess(warmup, embedding)
        self.assertLess(embedding, service_stop)

        embedding_check = result["checks"]["benchmark_comparison"]["comparisons"]["embedding"]
        self.assertEqual(embedding_check["warmup_command"]["name"], "benchmark_embedding_warmup")
        self.assertEqual(
            embedding_check["attempts"][0]["benchmark_command"]["command"],
            gate_module.approved_embedding_command(PROJECT_ROOT),
        )

    def test_gate_retries_failed_benchmark_comparison_with_unique_evidence(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            executor = FakeCommandExecutor(
                gate_module, fail_compare_attempts={"compare_operational": 2}
            )
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=executor,
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertTrue(result["ok"], msg=json.dumps(result, indent=2))
        operational = result["checks"]["benchmark_comparison"]["comparisons"]["operational"]
        self.assertEqual(operational["status"], "pass")
        self.assertEqual(len(operational["attempts"]), 3)
        self.assertEqual(operational["attempts"][0]["status"], "fail")
        self.assertEqual(operational["attempts"][1]["status"], "fail")
        self.assertEqual(operational["attempts"][2]["status"], "pass")
        self.assertEqual(
            executor.calls.count(gate_module.approved_operational_command(PROJECT_ROOT)), 3
        )
        attempt_stdout_paths = [
            attempt["benchmark_command"]["stdout_path"] for attempt in operational["attempts"]
        ]
        self.assertEqual(len(attempt_stdout_paths), len(set(attempt_stdout_paths)))

    def test_gate_computes_current_benchmark_assertions_from_comparisons(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = [
            "VAL-OPER-005",
            "VAL-OPER-011",
            "VAL-REPOBENCH-009",
            "VAL-CROSS-001",
            "VAL-CROSS-007",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(
                state_path,
                assertion_ids,
                pending={"VAL-OPER-005", "VAL-OPER-011", "VAL-REPOBENCH-009"},
            )
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=FakeCommandExecutor(gate_module),
                service_manager=FakeServiceManager(gate_module),
            )

            result = gate.run()

        self.assertTrue(result["ok"], msg=json.dumps(result, indent=2))
        matrix = {row["assertion_id"]: row for row in result["assertion_matrix"]}
        self.assertEqual(matrix["VAL-OPER-005"]["status"], "pass")
        self.assertEqual(matrix["VAL-OPER-011"]["status"], "pass")
        self.assertEqual(matrix["VAL-REPOBENCH-009"]["status"], "pass")
        self.assertEqual(matrix["VAL-CROSS-007"]["status"], "pass")

    def test_gate_stops_embedding_service_when_warmup_fails(self) -> None:
        gate_module = load_gate_script()
        assertion_ids = ["VAL-CROSS-001", "VAL-CROSS-004", "VAL-CROSS-007"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "validation-contract.md"
            state_path = root / "validation-state.json"
            write_contract(contract_path, assertion_ids)
            write_state(state_path, assertion_ids)
            service_manager = FakeServiceManager(gate_module)
            gate = gate_module.FinalQualityGate(
                project_root=PROJECT_ROOT,
                evidence_dir=root / "evidence",
                contract_path=contract_path,
                assertion_state_path=state_path,
                command_executor=FakeCommandExecutor(
                    gate_module, fail_command_names={"benchmark_embedding_warmup"}
                ),
                service_manager=service_manager,
            )

            result = gate.run()

        self.assertFalse(result["ok"])
        self.assertIn("embeddings", service_manager.stopped)
        self.assertEqual(result["checks"]["cleanup"]["status"], "pass")
        embedding = result["checks"]["benchmark_comparison"]["comparisons"]["embedding"]
        self.assertEqual(embedding["status"], "fail")
        self.assertIn("warmup", embedding["failure_reason"])


if __name__ == "__main__":
    unittest.main()
