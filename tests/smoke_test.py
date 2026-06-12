"""Smoke test installed Lode distributions."""

from __future__ import annotations

import subprocess
from importlib.metadata import version


def run_help(command: str) -> None:
    result = subprocess.run(
        [command, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "Lode" not in result.stdout and "lode" not in result.stdout:
        raise AssertionError(f"unexpected help output from {command!r}: {result.stdout!r}")


def main() -> None:
    if version("lode-kg") != "0.1.2":
        raise AssertionError("installed package version mismatch")

    import lode.cli  # noqa: F401
    import lode.daemon  # noqa: F401

    run_help("lode")
    run_help("loded")


if __name__ == "__main__":
    main()
