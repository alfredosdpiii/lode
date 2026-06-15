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
    def __init__(self, gate_module: Any, *, pollute_repo: Path | None = None) -> None:
        self._gate_module = gate_module
        self._pollute_repo = pollute_repo
        self.calls: list[str] = []

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
        if self._pollute_repo is not None and spec.name == "quality_format_check":
            (self._pollute_repo / "pollution.txt").write_text("generated", encoding="utf-8")

        stdout = ""
        if spec.name.startswith("benchmark_"):
            stdout = json.dumps({"ok": True, "parameters": {}})
        if spec.name.startswith("compare_"):
            stdout = json.dumps(
                {
                    "ok": True,
                    "overall_pass": True,
                    "metrics": {"fixture.metric": {"pass": True}},
                    "split_results": {},
                }
            )
        stdout_path = output_dir / f"{spec.name}.stdout"
        stderr_path = output_dir / f"{spec.name}.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return self._gate_module.CommandResult(
            name=spec.name,
            command=spec.command,
            exit_code=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            duration_seconds=0.01,
            stdout_preview=stdout,
            stderr_preview="",
        )


class FakeServiceManager:
    def __init__(self, gate_module: Any) -> None:
        self._gate_module = gate_module
        self.started: list[str] = []
        self.stopped: list[str] = []

    def ensure_started(self, name: str, evidence_dir: Path) -> Any:
        _ = evidence_dir
        self.started.append(name)
        return self._gate_module.ServiceRun(name=name, started_by_gate=True, ok=True, evidence=[])

    def stop_started(self, run: Any, evidence_dir: Path) -> Any:
        _ = evidence_dir
        self.stopped.append(run.name)
        return self._gate_module.ServiceStop(name=run.name, ok=True, evidence=[])


class FinalQualityGateTests(unittest.TestCase):
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
        self.assertEqual(
            executor.calls[: len(gate_module.QUALITY_COMMANDS)], gate_module.QUALITY_COMMANDS
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


if __name__ == "__main__":
    unittest.main()
