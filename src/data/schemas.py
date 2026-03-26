from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    text: str
    source: str
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTExample:
    messages: list[dict[str, str]]
    source: str
    example_id: str | None = None
    split_group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreferenceExample:
    prompt: list[dict[str, str]]
    chosen: str
    rejected: str
    source: str
    example_id: str | None = None
    split_group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
