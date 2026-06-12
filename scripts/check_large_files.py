from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAX_BYTES = int(os.environ.get("LODE_MAX_FILE_BYTES", "1000000"))


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def main() -> int:
    oversized = [
        path for path in tracked_files() if path.is_file() and path.stat().st_size > MAX_BYTES
    ]
    if oversized:
        for path in oversized:
            print(f"{path}: exceeds {MAX_BYTES} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
