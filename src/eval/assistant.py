from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation import resolve_stop_token_ids, strip_stop_strings
from tokenizer import SentencePieceTokenizer, format_chat


def _require_torch():
    import torch

    return torch


@dataclass(slots=True)
class AssistantBenchmarkResult:
    path: str
    examples: int
    average_score: float
    pass_rate: float
    exact_match_rate: float
    citation_rate: float


def _generate_response(
    model,
    tokenizer: SentencePieceTokenizer,
    prompt_messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    stop_strings: list[str],
) -> str:
    torch = _require_torch()
    rendered = format_chat(prompt_messages, add_generation_prompt=True)
    input_ids = torch.tensor(
        [tokenizer.encode(rendered, add_bos=True, add_eos=False)],
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        stop_token_ids=resolve_stop_token_ids(tokenizer, stop_strings),
    )
    new_tokens = generated[0, input_ids.size(1) :].tolist()
    return strip_stop_strings(tokenizer.decode(new_tokens).strip(), stop_strings)


def _looks_like_generic_filler(response: str) -> bool:
    response_lower = response.lower()
    filler_patterns = (
        "what would you like to work on",
        "what would you like help with",
        "how can i help today",
        "tell me what you want to figure out",
    )
    return any(pattern in response_lower for pattern in filler_patterns)


def _score_response(record: dict[str, Any], response: str, require_citations: bool) -> tuple[float, bool, bool]:
    score = 0.0
    exact = False
    if "reference" in record:
        exact = response.strip() == str(record["reference"]).strip()
        score += 1.0 if exact else 0.0
    if "expected_substrings" in record:
        matched = sum(1 for item in record["expected_substrings"] if item.lower() in response.lower())
        total = max(len(record["expected_substrings"]), 1)
        score += matched / total
    if "forbidden_substrings" in record:
        penalty = sum(1 for item in record["forbidden_substrings"] if item.lower() in response.lower())
        score -= penalty
    repeated_punctuation = re.search(r"([,.;:!?])(?:\s*\1){4,}", response)
    repeated_connector = re.search(r"\b(and|or)\b(?:\s+\1){3,}", response.lower())
    if repeated_punctuation or repeated_connector:
        score -= 2.0
    if _looks_like_generic_filler(response):
        score -= 1.0
    if "</s>" in response or "<|assistant|>" in response or "<|user|>" in response:
        score -= 2.0
    has_citation = "[source:" in response.lower() or "<|citation|>" in response.lower()
    if require_citations and record.get("requires_citation", False):
        score += 0.5 if has_citation else -0.5
    return max(score, 0.0), exact, has_citation


def run_assistant_benchmark(
    model,
    tokenizer_path: str,
    benchmark_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    stop_strings: list[str],
    require_citations: bool,
) -> AssistantBenchmarkResult:
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    rows = [json.loads(line) for line in Path(benchmark_path).read_text().splitlines() if line.strip()]
    scores: list[float] = []
    passes = 0
    exact_matches = 0
    citations = 0
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        response = _generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
            stop_strings,
        )
        score, exact, has_citation = _score_response(row, response, require_citations=require_citations)
        scores.append(score)
        passes += int(score >= float(row.get("pass_score", 1.0)))
        exact_matches += int(exact)
        citations += int(has_citation)
    total = max(len(rows), 1)
    return AssistantBenchmarkResult(
        path=benchmark_path,
        examples=len(rows),
        average_score=sum(scores) / total,
        pass_rate=passes / total,
        exact_match_rate=exact_matches / total,
        citation_rate=citations / total,
    )
