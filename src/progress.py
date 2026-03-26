from __future__ import annotations

import math
from dataclasses import dataclass


UNKNOWN_PERCENT_TEXT = "??.?%"
UNKNOWN_TIME_TEXT = "--:--:--"


@dataclass(slots=True)
class ProgressSnapshot:
    fraction_complete: float | None
    elapsed_seconds: float
    remaining_seconds: float | None
    summary: str


def _normalize_fraction(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return min(max(float(value), 0.0), 1.0)


def format_clock(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return UNKNOWN_TIME_TEXT
    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fraction_from_counts(*counts: tuple[int | float | None, int | float | None]) -> float | None:
    fractions: list[float] = []
    for completed, total in counts:
        if completed is None or total is None:
            continue
        total_value = float(total)
        if not math.isfinite(total_value) or total_value <= 0:
            continue
        completed_value = 0.0 if not math.isfinite(float(completed)) else float(completed)
        fractions.append(min(max(completed_value / total_value, 0.0), 1.0))
    if not fractions:
        return None
    return max(fractions)


def estimate_remaining_seconds(elapsed_seconds: float, fraction_complete: float | None) -> float | None:
    fraction = _normalize_fraction(fraction_complete)
    if fraction is None or fraction <= 0.0:
        return None
    if fraction >= 1.0:
        return 0.0
    return max(0.0, elapsed_seconds * (1.0 - fraction) / fraction)


def format_progress_summary(
    *,
    fraction_complete: float | None,
    elapsed_seconds: float,
    remaining_seconds: float | None = None,
) -> str:
    fraction = _normalize_fraction(fraction_complete)
    remaining = remaining_seconds if remaining_seconds is not None else estimate_remaining_seconds(
        elapsed_seconds,
        fraction,
    )
    percent_text = UNKNOWN_PERCENT_TEXT if fraction is None else f"{fraction * 100.0:.1f}%"
    return f"{percent_text} · {format_clock(elapsed_seconds)} elapsed · {format_clock(remaining)} left"


def build_progress_snapshot(
    elapsed_seconds: float,
    *counts: tuple[int | float | None, int | float | None],
) -> ProgressSnapshot:
    fraction = fraction_from_counts(*counts)
    remaining = estimate_remaining_seconds(elapsed_seconds, fraction)
    return ProgressSnapshot(
        fraction_complete=fraction,
        elapsed_seconds=elapsed_seconds,
        remaining_seconds=remaining,
        summary=format_progress_summary(
            fraction_complete=fraction,
            elapsed_seconds=elapsed_seconds,
            remaining_seconds=remaining,
        ),
    )
