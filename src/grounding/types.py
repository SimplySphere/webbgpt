from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Citation:
    source_type: str
    source_id: str
    label: str
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroundingHit:
    title: str
    content: str
    citations: list[Citation]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    route: str
    query: str
    grounded: bool
    school_years: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroundingResult:
    query: str
    hits: list[GroundingHit]
    route: str | None = None
    snapshot_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
