from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from typing import Any, TextIO

SENSITIVE_PARTS = ("password", "secret", "token", "key", "authorization")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in SENSITIVE_PARTS):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = sanitize(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    return value


def log_event(event: str, stream: TextIO | None = None, **fields: Any) -> None:
    payload = {
        "event": event,
        "time": round(time.time(), 3),
        **sanitize(fields),
    }
    print(json.dumps(payload, sort_keys=True), file=stream or sys.stderr)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[tuple[str, str, int]] = Counter()

    def record(self, method: str, path: str, status: int) -> None:
        with self._lock:
            self._counts[(method.upper(), path, status)] += 1

    def render(self) -> str:
        lines = [
            "# HELP loded_requests_total Total local daemon HTTP requests.",
            "# TYPE loded_requests_total counter",
        ]
        with self._lock:
            items = sorted(self._counts.items())
        for (method, path, status), count in items:
            lines.append(
                f'loded_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
        return "\n".join(lines) + "\n"
