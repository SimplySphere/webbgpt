from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable


SPECIAL_STOP_STRINGS = [
    "</s>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<|citation|>",
]


def default_stop_strings() -> list[str]:
    return list(SPECIAL_STOP_STRINGS)


def fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def resolve_stop_token_ids(tokenizer, stop_strings: Iterable[str]) -> list[int]:
    resolved: list[int] = []
    seen: set[int] = set()
    for value in stop_strings:
        try:
            token_id = int(tokenizer.token_to_id(value))
        except Exception:
            continue
        if token_id < 0 or token_id in seen:
            continue
        seen.add(token_id)
        resolved.append(token_id)
    eos_getter = getattr(tokenizer, "eos_token_id", None)
    if eos_getter is not None and eos_getter not in seen:
        resolved.append(int(eos_getter))
    return resolved


def strip_stop_strings(text: str, stop_strings: Iterable[str]) -> str:
    cleaned = text
    for stop in stop_strings:
        if not stop:
            continue
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]
    return cleaned.strip()


def apply_repetition_penalty(logits, generated_ids, repetition_penalty: float):
    if repetition_penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    batch_size = adjusted.size(0)
    for batch_index in range(batch_size):
        seen_tokens = set(int(token_id) for token_id in generated_ids[batch_index].tolist())
        for token_id in seen_tokens:
            value = adjusted[batch_index, token_id]
            adjusted[batch_index, token_id] = (
                value / repetition_penalty if value > 0 else value * repetition_penalty
            )
    return adjusted


def apply_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size: int):
    if no_repeat_ngram_size <= 1:
        return logits
    adjusted = logits.clone()
    batch_size, sequence_length = generated_ids.shape
    if sequence_length + 1 < no_repeat_ngram_size:
        return adjusted
    prefix_len = no_repeat_ngram_size - 1
    for batch_index in range(batch_size):
        sequence = [int(token_id) for token_id in generated_ids[batch_index].tolist()]
        if len(sequence) < prefix_len:
            continue
        prefix = tuple(sequence[-prefix_len:])
        banned_tokens: set[int] = set()
        ngram_map: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for start in range(len(sequence) - no_repeat_ngram_size + 1):
            key = tuple(sequence[start : start + prefix_len])
            next_token = sequence[start + prefix_len]
            ngram_map[key].add(next_token)
        banned_tokens = ngram_map.get(prefix, set())
        for token_id in banned_tokens:
            adjusted[batch_index, token_id] = float("-inf")
    return adjusted
