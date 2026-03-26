from __future__ import annotations

import re
from typing import Any


_SEPARATOR_BURST_RE = re.compile(r"([,./>\-])\1{2,}")
_TOKEN_RE = re.compile(r"\S+")


def analyze_generation(text: str) -> dict[str, Any]:
    stripped = text.strip()
    nonspace_chars = [char for char in stripped if not char.isspace()]
    nonspace_count = len(nonspace_chars)
    alpha_count = sum(char.isalpha() for char in nonspace_chars)
    alpha_ratio = 0.0 if nonspace_count == 0 else alpha_count / float(nonspace_count)
    comma_ratio = 0.0 if nonspace_count == 0 else stripped.count(",") / float(nonspace_count)
    separator_bursts = len(_SEPARATOR_BURST_RE.findall(stripped))

    raw_tokens = _TOKEN_RE.findall(stripped)
    normalized_tokens = [re.sub(r"[^a-z0-9]+", "", token.lower()) for token in raw_tokens]
    nonempty_normalized = [token for token in normalized_tokens if token]

    short_fragment_ratio = 0.0
    punctuation_suffix_ratio = 0.0
    unique_token_ratio = 0.0
    if raw_tokens:
        short_fragments = sum(1 for token in normalized_tokens if len(token) <= 2)
        punctuation_suffixes = sum(1 for token in raw_tokens if token[-1] in ",./->")
        short_fragment_ratio = short_fragments / float(len(raw_tokens))
        punctuation_suffix_ratio = punctuation_suffixes / float(len(raw_tokens))
    if nonempty_normalized:
        unique_token_ratio = len(set(nonempty_normalized)) / float(len(nonempty_normalized))

    repeated_token_run = 1
    current_run = 1
    previous = None
    for token in nonempty_normalized:
        if token == previous:
            current_run += 1
        else:
            current_run = 1
            previous = token
        repeated_token_run = max(repeated_token_run, current_run)

    reasons: list[str] = []
    if separator_bursts >= 2:
        reasons.append("separator_spam")
    if nonspace_count >= 40 and alpha_ratio < 0.45:
        reasons.append("low_alphabetic_ratio")
    if len(raw_tokens) >= 12 and short_fragment_ratio > 0.55:
        reasons.append("too_many_short_fragments")
    if len(raw_tokens) >= 12 and punctuation_suffix_ratio > 0.55:
        reasons.append("malformed_token_dump")
    if repeated_token_run >= 4:
        reasons.append("repeated_token_burst")
    if nonspace_count >= 40 and comma_ratio > 0.12:
        reasons.append("comma_spam")
    if len(nonempty_normalized) >= 12 and unique_token_ratio < 0.3:
        reasons.append("low_token_variety")

    degenerate = (
        len(reasons) >= 2
        or separator_bursts >= 4
        or repeated_token_run >= 6
        or (len(raw_tokens) >= 16 and punctuation_suffix_ratio > 0.7 and short_fragment_ratio > 0.6)
    )
    return {
        "degenerate": degenerate,
        "reasons": reasons,
        "metrics": {
            "nonspace_chars": nonspace_count,
            "alpha_ratio": round(alpha_ratio, 4),
            "comma_ratio": round(comma_ratio, 4),
            "separator_bursts": separator_bursts,
            "token_count": len(raw_tokens),
            "short_fragment_ratio": round(short_fragment_ratio, 4),
            "punctuation_suffix_ratio": round(punctuation_suffix_ratio, 4),
            "unique_token_ratio": round(unique_token_ratio, 4),
            "repeated_token_run": repeated_token_run,
        },
    }


def degenerate_output_message() -> str:
    return (
        "Response generation failed. The model produced malformed output. "
        "Retry, try Safe Retry, or open Debug Details to inspect the raw output."
    )
