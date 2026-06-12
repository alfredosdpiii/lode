from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARKER_RE = re.compile(r"\b(?:TODO|FIXME)\b(?!\((?:#\d+|[A-Z]+-\d+)\))")
SKIP_PATHS = {
    "scripts/check_todos.py",
    "uv.lock",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def main() -> int:
    violations: list[str] = []
    for path in tracked_files():
        if path.as_posix() in SKIP_PATHS or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if MARKER_RE.search(line):
                violations.append(f"{path}:{line_number}: TODO/FIXME must link an issue")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
