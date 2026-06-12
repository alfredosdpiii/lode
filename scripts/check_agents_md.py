from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_SNIPPETS = [
    "uv sync --extra kuzu --dev",
    "uv run ruff check .",
    "uv run mypy src tests scripts benchmarks",
    "uv run coverage run -m unittest discover -s tests -v",
    "uv run python scripts/check_large_files.py",
    "uv run python scripts/check_duplicate_code.py",
]


def main() -> int:
    path = Path("AGENTS.md")
    if not path.exists():
        print("AGENTS.md is missing", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        for snippet in missing:
            print(f"AGENTS.md missing command: {snippet}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
