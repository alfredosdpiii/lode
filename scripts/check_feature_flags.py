from __future__ import annotations

import ast
import sys
from pathlib import Path

FEATURES_FILE = Path("src/lode/features.py")


def configured_flags() -> set[str]:
    tree = ast.parse(FEATURES_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_DEFAULTS":
                    if isinstance(node.value, ast.Dict):
                        return {
                            key.value
                            for key in node.value.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)
                        }
    return set()


def main() -> int:
    flags = configured_flags()
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src").rglob("*.py")
        if path != FEATURES_FILE
    )
    dead = sorted(flag for flag in flags if flag not in text)
    if dead:
        for flag in dead:
            print(f"feature flag is not referenced: {flag}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
