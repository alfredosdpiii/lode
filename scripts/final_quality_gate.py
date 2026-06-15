#!/usr/bin/env python3
"""Final quality, benchmark, cleanup, and assertion-matrix gate for Lode."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUALITY_COMMANDS = [
    "uv run ruff format --check .",
    "uv run ruff check .",
    "uv run mypy src tests scripts benchmarks",
    "uv run coverage run -m unittest discover -s tests -v",
    "uv run coverage report",
    "uv run radon cc src tests scripts -s -n B",
    "uv run vulture src tests scripts --min-confidence 90",
    "uv run deptry .",
    "uv run bandit -c pyproject.toml -r src scripts -q",
    "uv run pip-audit --desc off",
    "uv run python scripts/check_large_files.py",
    "uv run python scripts/check_todos.py",
    "uv run python scripts/check_agents_md.py",
    "uv run python scripts/check_feature_flags.py",
    "uv run python scripts/check_duplicate_code.py",
    "uv run python scripts/generate_openapi_docs.py --check",
]

QUALITY_COMMAND_NAMES = [
    "quality_format_check",
    "quality_lint",
    "quality_typecheck",
    "quality_tests",
    "quality_coverage_report",
    "quality_complexity",
    "quality_dead_code",
    "quality_dependency_check",
    "quality_security",
    "quality_pip_audit",
    "quality_check_large_files",
    "quality_check_todos",
    "quality_check_agents_md",
    "quality_check_feature_flags",
    "quality_check_duplicate_code",
    "quality_openapi_check",
]

CLEARED_ENVIRONMENT_KEYS = {
    "LODE_EMBEDDINGS_URL",
    "KG_EMBEDDINGS_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "FACTORY_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}

HOSTED_URL_RE = re.compile(r"https?://(?!(?:127\.0\.0\.1|localhost|0\.0\.0\.0|embeddings)(?::|/))")
CREDENTIAL_NAME_RE = re.compile(r"\b(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)\b", re.IGNORECASE)
CONTRACT_ASSERTION_RE = re.compile(r"^###\s+(VAL-[A-Z]+-\d+):", re.MULTILINE)
MISSION_PROCESS_RE = re.compile(r"\b(loded|lode serve|bench_lode\.py|repobench_adapter\.py)\b")
TEMP_DIR_PREFIXES = ("lode-bench-", "lode-repobench-", "lode-validation-")


def approved_operational_command(project_root: Path = PROJECT_ROOT) -> str:
    return (
        "uv run python scripts/bench_lode.py "
        f"--repo {shlex.quote(str(project_root.resolve()))} "
        '--repeat 10 --limit 20 --budget 4000 --query "build context pack" '
        '--query "embedding queue" --symbol build_context_pack --include-kuzu --json'
    )


def approved_embedding_command(project_root: Path = PROJECT_ROOT) -> str:
    return (
        "LODE_EMBEDDINGS_MODEL=Snowflake/snowflake-arctic-embed-s "
        "uv run python scripts/bench_lode.py "
        f"--repo {shlex.quote(str(project_root.resolve()))} "
        '--repeat 10 --limit 20 --budget 4000 --query "build context pack" '
        '--query "embedding queue" --symbol build_context_pack '
        "--embed-url http://127.0.0.1:7980 --embed-limit 32 --json"
    )


def approved_repobench_command(_project_root: Path = PROJECT_ROOT) -> str:
    return (
        "uv run python benchmarks/repobench_adapter.py "
        "--input /home/bryan/.cache/lode/benchmarks/repobench_python_v1.1/jsonl "
        "--mode context --top-k 1 3 5 10 --query-lines 5 --search-limit 30 "
        "--context-budget 6000 --json"
    )


@dataclass
class CommandSpec:
    name: str
    command: str
    argv: list[str]
    env_overrides: dict[str, str] | None = None
    timeout: int | None = None
    local_first: bool = True
    scan_for_hosted_endpoints: bool = False


@dataclass
class CommandResult:
    name: str
    command: str
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    stdout_preview: str
    stderr_preview: str


@dataclass
class ServiceRun:
    name: str
    started_by_gate: bool
    ok: bool
    evidence: list[dict[str, Any]]
    failure_reason: str = ""


@dataclass
class ServiceStop:
    name: str
    ok: bool
    evidence: list[dict[str, Any]]
    failure_reason: str = ""


@dataclass
class ServiceSpec:
    name: str
    start: str
    stop: str
    healthcheck: str
    port: int | None


class SubprocessCommandExecutor:
    def run(
        self,
        spec: CommandSpec,
        *,
        cwd: Path,
        env: dict[str, str],
        output_dir: Path,
        timeout: int | None,
    ) -> CommandResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stdout = ""
        stderr = ""
        exit_code = 0
        try:
            process = subprocess.Popen(
                spec.argv,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = int(process.returncode)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stderr = f"{stderr}\ncommand timed out after {timeout} seconds".strip()
            exit_code = 124
        except OSError as exc:
            stderr = str(exc)
            exit_code = 127
        duration = time.monotonic() - started
        stdout_path = output_dir / f"{spec.name}.stdout"
        stderr_path = output_dir / f"{spec.name}.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return CommandResult(
            name=spec.name,
            command=spec.command,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            duration_seconds=round(duration, 3),
            stdout_preview=stdout[:4000],
            stderr_preview=stderr[:4000],
        )


class ManifestServiceManager:
    def __init__(
        self,
        services_path: Path | None,
        command_executor: SubprocessCommandExecutor | None = None,
    ) -> None:
        self._services = load_services(services_path) if services_path else {}
        self._executor = command_executor or SubprocessCommandExecutor()

    def ensure_started(self, name: str, evidence_dir: Path) -> ServiceRun:
        spec = self._services.get(name)
        if spec is None:
            return ServiceRun(
                name=name,
                started_by_gate=False,
                ok=False,
                evidence=[],
                failure_reason=f"service {name!r} is not declared in services.yaml",
            )

        evidence: list[dict[str, Any]] = []
        if spec.port is not None and is_port_open(spec.port):
            health = self._run_manifest_command(
                f"service_{name}_preexisting_healthcheck",
                spec.healthcheck,
                evidence_dir,
                timeout=30,
            )
            evidence.append(command_result_record(health))
            return ServiceRun(
                name=name,
                started_by_gate=False,
                ok=health.exit_code == 0,
                evidence=evidence,
                failure_reason=""
                if health.exit_code == 0
                else f"pre-existing {name} failed healthcheck",
            )

        start = self._run_manifest_command(
            f"service_{name}_start", spec.start, evidence_dir, timeout=300
        )
        evidence.append(command_result_record(start))
        if start.exit_code != 0:
            return ServiceRun(
                name=name,
                started_by_gate=True,
                ok=False,
                evidence=evidence,
                failure_reason=f"service {name} start command exited {start.exit_code}",
            )

        ok = False
        for attempt in range(1, 11):
            time.sleep(min(attempt, 5))
            health = self._run_manifest_command(
                f"service_{name}_healthcheck_{attempt}",
                spec.healthcheck,
                evidence_dir,
                timeout=30,
            )
            evidence.append(command_result_record(health))
            if health.exit_code == 0:
                ok = True
                break
        return ServiceRun(
            name=name,
            started_by_gate=True,
            ok=ok,
            evidence=evidence,
            failure_reason="" if ok else f"service {name} healthcheck failed after start",
        )

    def stop_started(self, run: ServiceRun, evidence_dir: Path) -> ServiceStop:
        if not run.started_by_gate:
            return ServiceStop(name=run.name, ok=True, evidence=[])
        spec = self._services.get(run.name)
        if spec is None:
            return ServiceStop(
                name=run.name,
                ok=False,
                evidence=[],
                failure_reason=f"service {run.name!r} is not declared in services.yaml",
            )
        stop = self._run_manifest_command(
            f"service_{run.name}_stop", spec.stop, evidence_dir, timeout=120
        )
        return ServiceStop(
            name=run.name,
            ok=stop.exit_code == 0,
            evidence=[command_result_record(stop)],
            failure_reason="" if stop.exit_code == 0 else f"service {run.name} stop failed",
        )

    def _run_manifest_command(
        self, name: str, command: str, evidence_dir: Path, *, timeout: int
    ) -> CommandResult:
        spec = CommandSpec(
            name=name,
            command=command,
            argv=["sh", "-c", command],
            timeout=timeout,
            local_first=False,
        )
        env, _cleared = build_command_env(
            overrides={"LODE_VALIDATION_RUN_ID": validation_run_id()}, local_first=False
        )
        return self._executor.run(
            spec, cwd=PROJECT_ROOT, env=env, output_dir=evidence_dir / "services", timeout=timeout
        )


class FinalQualityGate:
    def __init__(
        self,
        *,
        project_root: Path,
        evidence_dir: Path,
        contract_path: Path,
        assertion_state_path: Path,
        services_path: Path | None = None,
        command_executor: Any | None = None,
        service_manager: Any | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.evidence_dir = evidence_dir.resolve()
        self.contract_path = contract_path
        self.assertion_state_path = assertion_state_path
        self.command_executor = command_executor or SubprocessCommandExecutor()
        self.service_manager = service_manager or ManifestServiceManager(services_path)
        self.command_results: list[CommandResult] = []
        self.service_runs: list[ServiceRun] = []
        self.service_stops: list[ServiceStop] = []
        self.artifacts: list[Path] = []
        self.cleared_environment_keys: set[str] = set()

    def run(self) -> dict[str, Any]:
        self._prepare_evidence_dir()
        before = capture_snapshot("before", self.project_root, self.evidence_dir)
        self.artifacts.append(Path(before["artifact_path"]))

        benchmark_status: dict[str, Any] = {
            "status": "fail",
            "comparisons": {},
            "failure_reason": "benchmark comparisons did not run",
        }
        quality_status: dict[str, Any] = {"status": "fail", "commands": []}
        local_first_status: dict[str, Any] = {"status": "fail"}
        cleanup_status: dict[str, Any] = {"status": "fail"}
        artifact_status: dict[str, Any] = {"status": "fail"}

        try:
            quality_status = self._run_quality_commands()
            benchmark_status = self._run_benchmark_comparisons()
        finally:
            self._stop_started_services()
            after = capture_snapshot("after", self.project_root, self.evidence_dir)
            self.artifacts.append(Path(after["artifact_path"]))
            cleanup_status = compare_cleanup(before, after, self.service_stops)
            artifact_status = compare_artifact_hygiene(before, after, self.evidence_dir)
            local_first_status = self._check_local_first()

        checks = {
            "quality_gate": quality_status,
            "benchmark_comparison": benchmark_status,
            "cleanup": cleanup_status,
            "artifact_hygiene": artifact_status,
        }
        final_artifacts = dedupe_paths(
            [
                *self.artifacts,
                self.evidence_dir / "assertion-matrix.json",
                self.evidence_dir / "final-quality-gate.json",
            ]
        )
        final_artifact_strings = [str(path) for path in final_artifacts]
        assertion_matrix = build_assertion_matrix(
            contract_path=self.contract_path,
            state_path=self.assertion_state_path,
            checks=checks,
            local_first=local_first_status,
            artifacts=final_artifact_strings,
        )

        result = {
            "ok": all(row["status"] == "pass" for row in assertion_matrix),
            "project_root": str(self.project_root),
            "evidence_dir": str(self.evidence_dir),
            "checks": checks,
            "local_first": local_first_status,
            "assertion_matrix": assertion_matrix,
            "commands": [command_result_record(result) for result in self.command_results],
            "services": {
                "runs": [asdict(run) for run in self.service_runs],
                "stops": [asdict(stop) for stop in self.service_stops],
            },
            "artifacts": final_artifact_strings,
        }
        self._write_json("assertion-matrix.json", assertion_matrix)
        self._write_json("final-quality-gate.json", result)
        return result

    def _prepare_evidence_dir(self) -> None:
        if is_relative_to(self.evidence_dir, self.project_root):
            raise ValueError(
                f"evidence directory must be outside the repository: {self.evidence_dir}"
            )
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _run_quality_commands(self) -> dict[str, Any]:
        records = []
        for name, command in zip(QUALITY_COMMAND_NAMES, QUALITY_COMMANDS, strict=True):
            spec = CommandSpec(
                name=name,
                command=command,
                argv=shlex.split(command),
                timeout=1800,
                local_first=True,
            )
            result = self._run_command(spec, output_subdir="quality")
            records.append(command_result_record(result))
        failed = [record for record in records if record["exit_code"] != 0]
        return {
            "status": "pass" if not failed and len(records) == len(QUALITY_COMMANDS) else "fail",
            "commands": records,
            "failure_reason": "" if not failed else f"{len(failed)} quality command(s) failed",
        }

    def _run_benchmark_comparisons(self) -> dict[str, Any]:
        comparisons: dict[str, Any] = {}
        benchmark_specs = [
            self._operational_benchmark_spec(),
            self._embedding_benchmark_spec(),
            self._repobench_benchmark_spec(),
        ]

        embedding_service: ServiceRun | None = None
        for spec in benchmark_specs:
            if spec.name == "benchmark_embeddings":
                embedding_service = self.service_manager.ensure_started(
                    "embeddings", self.evidence_dir
                )
                self.service_runs.append(embedding_service)
                self._add_service_artifacts(embedding_service.evidence)
                if not embedding_service.ok:
                    comparisons["embedding"] = {
                        "status": "fail",
                        "failure_reason": embedding_service.failure_reason,
                    }
                    continue
            benchmark_result = self._run_command(spec, output_subdir="benchmarks")
            comparison_type = spec.name.removeprefix("benchmark_")
            if comparison_type == "embeddings":
                comparison_type = "embedding"
            comparisons[comparison_type] = self._compare_benchmark(
                comparison_type, benchmark_result, spec.command
            )
            if embedding_service is not None and spec.name == "benchmark_embeddings":
                stop = self.service_manager.stop_started(embedding_service, self.evidence_dir)
                self.service_stops.append(stop)
                self._add_service_artifacts(stop.evidence)
                embedding_service = None

        if embedding_service is not None:
            stop = self.service_manager.stop_started(embedding_service, self.evidence_dir)
            self.service_stops.append(stop)
            self._add_service_artifacts(stop.evidence)

        failed = {
            name: comparison
            for name, comparison in comparisons.items()
            if comparison.get("status") != "pass"
        }
        missing = sorted({"operational", "embedding", "repobench"} - set(comparisons))
        return {
            "status": "pass" if not failed and not missing else "fail",
            "comparisons": comparisons,
            "missing_comparisons": missing,
            "failure_reason": (
                "" if not failed and not missing else "one or more benchmark comparisons failed"
            ),
        }

    def _operational_benchmark_spec(self) -> CommandSpec:
        command = approved_operational_command(self.project_root)
        return CommandSpec(
            name="benchmark_operational",
            command=command,
            argv=shlex.split(command),
            timeout=1800,
            local_first=True,
            scan_for_hosted_endpoints=True,
        )

    def _embedding_benchmark_spec(self) -> CommandSpec:
        command = approved_embedding_command(self.project_root)
        execution_command = command.removeprefix(
            "LODE_EMBEDDINGS_MODEL=Snowflake/snowflake-arctic-embed-s "
        )
        return CommandSpec(
            name="benchmark_embeddings",
            command=command,
            argv=shlex.split(execution_command),
            env_overrides={"LODE_EMBEDDINGS_MODEL": "Snowflake/snowflake-arctic-embed-s"},
            timeout=1800,
            local_first=True,
            scan_for_hosted_endpoints=True,
        )

    def _repobench_benchmark_spec(self) -> CommandSpec:
        command = approved_repobench_command(self.project_root)
        return CommandSpec(
            name="benchmark_repobench",
            command=command,
            argv=shlex.split(command),
            timeout=3600,
            local_first=True,
            scan_for_hosted_endpoints=True,
        )

    def _compare_benchmark(
        self, comparison_type: str, benchmark_result: CommandResult, benchmark_command: str
    ) -> dict[str, Any]:
        if benchmark_result.exit_code != 0:
            return {
                "status": "fail",
                "benchmark_command": command_result_record(benchmark_result),
                "failure_reason": f"benchmark command exited {benchmark_result.exit_code}",
            }
        output_path = self.evidence_dir / "comparisons" / f"{comparison_type}.json"
        argv = [
            "uv",
            "run",
            "python",
            "scripts/bench_compare.py",
            "--type",
            comparison_type,
            "--current",
            str(benchmark_result.stdout_path),
            "--command",
            benchmark_command,
            "--output",
            str(output_path),
        ]
        if comparison_type == "embedding":
            argv.extend(["--env", "LODE_EMBEDDINGS_MODEL=Snowflake/snowflake-arctic-embed-s"])
        command = shlex.join(argv)
        compare_spec = CommandSpec(
            name=f"compare_{comparison_type}",
            command=command,
            argv=argv,
            timeout=300,
            local_first=True,
            scan_for_hosted_endpoints=True,
        )
        compare_result = self._run_command(compare_spec, output_subdir="comparisons")
        parsed = load_json_file(output_path if output_path.exists() else compare_result.stdout_path)
        overall_pass = bool(parsed.get("overall_pass")) if isinstance(parsed, dict) else False
        return {
            "status": "pass" if compare_result.exit_code == 0 and overall_pass else "fail",
            "benchmark_command": command_result_record(benchmark_result),
            "compare_command": command_result_record(compare_result),
            "comparison_artifact": str(output_path),
            "overall_pass": overall_pass,
            "failure_reason": (
                ""
                if compare_result.exit_code == 0 and overall_pass
                else f"comparison exited {compare_result.exit_code} overall_pass={overall_pass}"
            ),
        }

    def _run_command(self, spec: CommandSpec, *, output_subdir: str) -> CommandResult:
        env, cleared = build_command_env(
            overrides=spec.env_overrides or {}, local_first=spec.local_first
        )
        self.cleared_environment_keys.update(cleared)
        output_dir = self.evidence_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.command_executor.run(
            spec,
            cwd=self.project_root,
            env=env,
            output_dir=output_dir,
            timeout=spec.timeout,
        )
        self.command_results.append(result)
        self.artifacts.extend([result.stdout_path, result.stderr_path])
        return result

    def _stop_started_services(self) -> None:
        stopped = {stop.name for stop in self.service_stops}
        for run in self.service_runs:
            if run.started_by_gate and run.name not in stopped:
                stop = self.service_manager.stop_started(run, self.evidence_dir)
                self.service_stops.append(stop)
                self._add_service_artifacts(stop.evidence)

    def _check_local_first(self) -> dict[str, Any]:
        scanned = []
        violations = []
        for result in self.command_results:
            if not result.name.startswith(("benchmark_", "compare_")):
                continue
            for path in (result.stdout_path, result.stderr_path):
                text = safe_read_text(path)
                scanned.append(str(path))
                if HOSTED_URL_RE.search(text):
                    violations.append(f"hosted URL detected in {path}")
                if CREDENTIAL_NAME_RE.search(text) and "LODE_EMBEDDINGS_MODEL" not in text:
                    violations.append(f"credential-like token name detected in {path}")
        return {
            "status": "pass" if not violations else "fail",
            "cleared_environment_keys": sorted(self.cleared_environment_keys),
            "scanned_artifacts": scanned,
            "violations": violations,
            "failure_reason": "" if not violations else "; ".join(violations),
        }

    def _write_json(self, filename: str, payload: Any) -> Path:
        path = self.evidence_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if path not in self.artifacts:
            self.artifacts.append(path)
        return path

    def _add_service_artifacts(self, evidence: list[dict[str, Any]]) -> None:
        for record in evidence:
            for key in ("stdout_path", "stderr_path"):
                value = record.get(key)
                if value:
                    self.artifacts.append(Path(value))


def build_command_env(
    *, overrides: dict[str, str] | None = None, local_first: bool = True
) -> tuple[dict[str, str], set[str]]:
    env = os.environ.copy()
    cleared: set[str] = set()
    if local_first:
        for key in list(env):
            if key in CLEARED_ENVIRONMENT_KEYS:
                env.pop(key, None)
                cleared.add(key)
        cleared.update(CLEARED_ENVIRONMENT_KEYS)
    for key, value in (overrides or {}).items():
        env[key] = value
    return env, cleared


def capture_snapshot(label: str, project_root: Path, evidence_dir: Path) -> dict[str, Any]:
    snapshot = {
        "label": label,
        "git_status": git_status(project_root),
        "allowed_untracked_fingerprints": {
            rel: fingerprint_tree(project_root / rel) for rel in (".factory", "droid-wiki")
        },
        "ports": {str(port): port_snapshot(port) for port in (7979, 7980)},
        "processes": process_snapshot(),
        "docker": docker_snapshot(),
        "temp_dirs": temp_dir_snapshot(),
    }
    path = evidence_dir / f"snapshot-{label}.json"
    snapshot["artifact_path"] = str(path)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def git_status(project_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "exit_code": result.returncode,
        "lines": result.stdout.splitlines(),
        "stderr": result.stderr,
    }


def fingerprint_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "entries": []}
    entries = []
    for child in sorted(path.rglob("*")):
        try:
            stat = child.stat()
        except OSError:
            continue
        entries.append(
            {
                "path": str(child.relative_to(path.parent)),
                "is_dir": child.is_dir(),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return {"exists": True, "entries": entries}


def port_snapshot(port: int) -> dict[str, Any]:
    ss_result = subprocess.run(
        ["ss", "-ltn", "sport", "=", f":{port}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return {
        "listening": is_port_open(port),
        "ss_exit_code": ss_result.returncode,
        "ss_stdout": ss_result.stdout,
        "ss_stderr": ss_result.stderr,
    }


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def process_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if MISSION_PROCESS_RE.search(line)]
    return {"exit_code": result.returncode, "lines": lines, "stderr": result.stderr}


def docker_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Status}}"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if "lode-validation" in line]
    return {"exit_code": result.returncode, "lines": lines, "stderr": result.stderr}


def temp_dir_snapshot() -> list[str]:
    temp_root = Path(tempfile.gettempdir())
    dirs: list[str] = []
    for prefix in TEMP_DIR_PREFIXES:
        dirs.extend(str(path) for path in temp_root.glob(f"{prefix}*") if path.exists())
    return sorted(dirs)


def compare_cleanup(
    before: dict[str, Any], after: dict[str, Any], service_stops: list[ServiceStop] | None = None
) -> dict[str, Any]:
    issues = []
    if before["ports"] != after["ports"]:
        issues.append("port snapshot changed for 7979 or 7980")

    before_processes = set(before["processes"].get("lines", []))
    after_processes = set(after["processes"].get("lines", []))
    new_processes = sorted(after_processes - before_processes)
    if new_processes:
        issues.append(f"mission process snapshot has new entries: {new_processes}")

    before_containers = set(before["docker"].get("lines", []))
    after_containers = set(after["docker"].get("lines", []))
    new_containers = sorted(after_containers - before_containers)
    if new_containers:
        issues.append(f"mission container snapshot has new entries: {new_containers}")

    before_temp_dirs = set(before.get("temp_dirs", []))
    after_temp_dirs = set(after.get("temp_dirs", []))
    new_temp_dirs = sorted(after_temp_dirs - before_temp_dirs)
    if new_temp_dirs:
        issues.append(f"mission temp directories remain: {new_temp_dirs}")

    for stop in service_stops or []:
        if not stop.ok:
            reason = f": {stop.failure_reason}" if stop.failure_reason else ""
            issues.append(f"service {stop.name} cleanup failed{reason}")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "failure_reason": "" if not issues else "; ".join(issues),
    }


def compare_artifact_hygiene(
    before: dict[str, Any], after: dict[str, Any], evidence_dir: Path
) -> dict[str, Any]:
    issues = []
    before_status = before["git_status"]
    after_status = after["git_status"]
    if before_status != after_status:
        issues.append("git status changed during validation")
    if before["allowed_untracked_fingerprints"] != after["allowed_untracked_fingerprints"]:
        issues.append("pre-existing untracked .factory or droid-wiki metadata changed")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "failure_reason": "" if not issues else "; ".join(issues),
        "evidence_listing": sorted(str(path) for path in evidence_dir.rglob("*") if path.is_file()),
    }


def build_assertion_matrix(
    *,
    contract_path: Path,
    state_path: Path,
    checks: dict[str, Any],
    local_first: dict[str, Any],
    artifacts: list[str],
) -> list[dict[str, Any]]:
    assertion_ids = parse_assertion_ids(contract_path)
    assertion_state = load_assertion_state(state_path)
    current_rows = {
        "VAL-CROSS-001": row_from_check(
            "VAL-CROSS-001",
            "quality gate and benchmark comparison",
            checks["quality_gate"]["status"] == "pass"
            and checks["benchmark_comparison"]["status"] == "pass",
            "; ".join(
                reason
                for reason in (
                    checks["quality_gate"].get("failure_reason", ""),
                    checks["benchmark_comparison"].get("failure_reason", ""),
                )
                if reason
            ),
            artifacts,
        ),
        "VAL-CROSS-004": row_from_check(
            "VAL-CROSS-004",
            "cleanup snapshot",
            checks["cleanup"]["status"] == "pass",
            checks["cleanup"].get("failure_reason", ""),
            artifacts,
        ),
        "VAL-CROSS-005": row_from_check(
            "VAL-CROSS-005",
            "git status and artifact hygiene snapshot",
            checks["artifact_hygiene"]["status"] == "pass",
            checks["artifact_hygiene"].get("failure_reason", ""),
            artifacts,
        ),
        "VAL-CROSS-006": row_from_check(
            "VAL-CROSS-006",
            "local-first sanitized environment and log scan",
            local_first["status"] == "pass",
            local_first.get("failure_reason", ""),
            artifacts,
        ),
    }

    rows = []
    for assertion_id in assertion_ids:
        if assertion_id == "VAL-CROSS-007":
            continue
        if assertion_id in current_rows:
            rows.append(current_rows[assertion_id])
        else:
            rows.append(row_from_state(assertion_id, assertion_state.get(assertion_id), state_path))

    final_pass = all(row["status"] == "pass" for row in rows)
    rows.append(
        row_from_check(
            "VAL-CROSS-007",
            "final assertion matrix integrity",
            final_pass,
            "" if final_pass else "one or more assertion matrix rows failed",
            artifacts,
        )
    )
    present = {row["assertion_id"] for row in rows}
    for assertion_id in assertion_ids:
        if assertion_id not in present:
            rows.append(
                row_from_check(
                    assertion_id,
                    "final assertion matrix integrity",
                    False,
                    "assertion was missing from matrix construction",
                    artifacts,
                )
            )
    return sorted(rows, key=lambda row: assertion_ids.index(row["assertion_id"]))


def parse_assertion_ids(contract_path: Path) -> list[str]:
    text = contract_path.read_text(encoding="utf-8")
    seen: set[str] = set()
    assertion_ids = []
    for match in CONTRACT_ASSERTION_RE.finditer(text):
        assertion_id = match.group(1)
        if assertion_id not in seen:
            assertion_ids.append(assertion_id)
            seen.add(assertion_id)
    if not assertion_ids:
        raise ValueError(f"no validation assertion IDs found in {contract_path}")
    return assertion_ids


def load_assertion_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assertions = data.get("assertions", {})
    return assertions if isinstance(assertions, dict) else {}


def row_from_state(assertion_id: str, state: Any, state_path: Path) -> dict[str, Any]:
    if not isinstance(state, dict):
        return row_from_check(
            assertion_id,
            "validation state",
            False,
            f"assertion is missing from {state_path}",
            [str(state_path)],
        )
    status = state.get("status")
    passed = status in {"pass", "passed"}
    reason = "" if passed else f"assertion state is {status or 'missing'}"
    return {
        "assertion_id": assertion_id,
        "status": "pass" if passed else "fail",
        "command_or_tool": state.get("commandOrTool")
        or state.get("validatedAtMilestone")
        or "validation state",
        "artifact_paths": state.get("artifactPaths", [str(state_path)]),
        "failure_reason": reason,
    }


def row_from_check(
    assertion_id: str,
    command_or_tool: str,
    passed: bool,
    failure_reason: str,
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "assertion_id": assertion_id,
        "status": "pass" if passed else "fail",
        "command_or_tool": command_or_tool,
        "artifact_paths": artifacts,
        "failure_reason": "" if passed else failure_reason,
    }


def load_services(path: Path | None) -> dict[str, ServiceSpec]:
    if path is None or not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    services: dict[str, dict[str, str]] = {}
    in_services = False
    current: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "services:":
            in_services = True
            index += 1
            continue
        if in_services and line and not line.startswith(" "):
            break
        if in_services:
            service_match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
            if service_match:
                current = service_match.group(1)
                services[current] = {}
                index += 1
                continue
            field_match = re.match(r"^    ([A-Za-z0-9_-]+):\s*(.*)$", line)
            if field_match and current:
                key, value = field_match.group(1), field_match.group(2).strip()
                if value in {">-", ">", "|", "|-"}:
                    block: list[str] = []
                    index += 1
                    while index < len(lines):
                        block_line = lines[index]
                        if re.match(r"^    [A-Za-z0-9_-]+:\s*", block_line) or re.match(
                            r"^  [A-Za-z0-9_-]+:\s*$", block_line
                        ):
                            index -= 1
                            break
                        if block_line.startswith("      "):
                            block.append(block_line.strip())
                        index += 1
                    services[current][key] = " ".join(block)
                else:
                    services[current][key] = value.strip("'\"")
        index += 1

    specs = {}
    for name, values in services.items():
        if {"start", "stop", "healthcheck"} <= values.keys():
            port_value = values.get("port")
            specs[name] = ServiceSpec(
                name=name,
                start=values["start"],
                stop=values["stop"],
                healthcheck=values["healthcheck"],
                port=int(port_value) if port_value and port_value.isdigit() else None,
            )
    return specs


def command_result_record(result: CommandResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
        "duration_seconds": result.duration_seconds,
        "stdout_preview": result.stdout_preview,
        "stderr_preview": result.stderr_preview,
    }


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def validation_run_id() -> str:
    return os.environ.get("LODE_VALIDATION_RUN_ID", f"gate-{os.getpid()}")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final Lode quality, benchmark, cleanup, and assertion gate."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mission-dir", type=Path)
    parser.add_argument("--validation-contract", type=Path)
    parser.add_argument("--assertion-state", type=Path)
    parser.add_argument("--services-file", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mission_dir = args.mission_dir
    contract_path = args.validation_contract or (
        mission_dir / "validation-contract.md" if mission_dir else None
    )
    assertion_state_path = args.assertion_state or (
        mission_dir / "validation-state.json" if mission_dir else None
    )
    services_path = args.services_file or (mission_dir / "services.yaml" if mission_dir else None)
    evidence_dir = args.evidence_dir or Path(tempfile.mkdtemp(prefix="lode-final-quality-gate-"))
    if contract_path is None or assertion_state_path is None:
        parser.error(
            "--mission-dir or both --validation-contract and --assertion-state are required"
        )
    gate = FinalQualityGate(
        project_root=args.project_root,
        evidence_dir=evidence_dir,
        contract_path=contract_path,
        assertion_state_path=assertion_state_path,
        services_path=services_path,
    )
    result = gate.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
