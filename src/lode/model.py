from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Node:
    id: str
    kind: str
    name: str
    qname: str
    path: str
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    doc: str = ""
    confidence: str = "strong"
    content_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    kind: str
    confidence: str = "strong"
    detail: str = ""


@dataclass(slots=True)
class FileIndex:
    path: str
    abspath: str
    language: str
    size: int
    content_hash: str
    generated: bool
    nodes: list[Node]
    edges: list[Edge]
