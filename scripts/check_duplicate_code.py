from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MIN_LINES = 12


def normalized_blocks(path: Path) -> dict[str, int]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    blocks: dict[str, int] = {}
    for index in range(0, max(0, len(lines) - MIN_LINES + 1)):
        block = "\n".join(lines[index : index + MIN_LINES])
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        blocks[digest] = index + 1
    return blocks


def main() -> int:
    seen: dict[str, tuple[Path, int]] = {}
    duplicates: list[str] = []
    for path in [*Path("src").rglob("*.py"), *Path("scripts").rglob("*.py")]:
        for digest, line in normalized_blocks(path).items():
            previous = seen.get(digest)
            if previous:
                old_path, old_line = previous
                duplicates.append(f"{path}:{line} duplicates {old_path}:{old_line}")
            else:
                seen[digest] = (path, line)
    if duplicates:
        print("\n".join(duplicates), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
