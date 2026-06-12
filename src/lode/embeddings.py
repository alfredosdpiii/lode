from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


def embeddings_url() -> str | None:
    return os.environ.get("LODE_EMBEDDINGS_URL") or os.environ.get("KG_EMBEDDINGS_URL")


def embeddings_model() -> str:
    return (
        os.environ.get("LODE_EMBEDDINGS_MODEL")
        or os.environ.get("KG_EMBEDDINGS_MODEL")
        or "unknown"
    )


def embed_texts(texts: list[str], url: str | None = None, retries: int = 2) -> list[list[float]]:
    endpoint = (url or embeddings_url() or "").rstrip("/")
    if not endpoint:
        raise RuntimeError("No local embeddings endpoint configured. Set LODE_EMBEDDINGS_URL.")
    payload = json.dumps({"inputs": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/embed",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    data: Any = None
    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError("Embedding endpoint failed after retries") from exc
            time.sleep(0.25 * (2**attempt))
    else:
        raise RuntimeError("Embedding endpoint failed") from last_error
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    raise RuntimeError("Unexpected embeddings response shape")
