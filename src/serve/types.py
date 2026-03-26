from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grounding.types import Citation


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class AssistantResponse:
    text: str
    used_tools: bool
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

