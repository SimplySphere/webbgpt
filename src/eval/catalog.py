from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generation import resolve_stop_token_ids, strip_stop_strings
from tokenizer import SentencePieceTokenizer
from tokenizer import format_chat


def _require_torch():
    import torch

    return torch


@dataclass(slots=True)
class CatalogBenchmarkResult:
    path: str
    examples: int
    exactness: float
    citation_rate: float
    abstention_rate: float
    attribution_lanes: dict[str, dict[str, float]] = field(default_factory=dict)
    retrieval_audit: dict[str, Any] = field(default_factory=dict)


class _EvalModelBackend:
    def __init__(self, model, tokenizer_path: str):
        self.model = model
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        stop_strings: list[str],
    ) -> str:
        torch = _require_torch()
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_bos=True, add_eos=False)],
            dtype=torch.long,
            device=self.model.device,
        )
        attention_mask = torch.ones_like(input_ids)
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_token_ids=resolve_stop_token_ids(self.tokenizer, stop_strings),
        )
        new_tokens = generated[0, input_ids.size(1) :].tolist()
        return strip_stop_strings(self.tokenizer.decode(new_tokens).strip(), stop_strings)


def _did_abstain(response_lower: str) -> bool:
    abstention_markers = (
        "i don't know",
        "i do not know",
        "not enough information",
        "insufficient information",
        "i cannot confirm",
        "i can't confirm",
        "not in the catalog",
        "not listed in the catalog",
        "could not find",
        "couldn't find",
        "no matching catalog entries",
    )
    return any(marker in response_lower for marker in abstention_markers)


def score_catalog_response(record: dict[str, Any], response: str) -> tuple[float, bool, bool]:
    response_lower = response.lower()
    has_citation = "[source:" in response_lower or "<|citation|>" in response_lower
    abstained = _did_abstain(response_lower)
    expected = [item.lower() for item in record.get("expected_course_codes", [])]
    expects_abstention = bool(record.get("expects_abstention", False))
    if expects_abstention:
        exact = abstained
    elif not expected:
        exact = bool(response.strip())
    else:
        exact = all(item in response_lower for item in expected)
    return float(exact), has_citation, abstained


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    return next(
        (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
        messages[-1].get("content", "") if messages else "",
    )


def _oracle_response(query: str, grounding_provider) -> tuple[str, Any]:
    result = grounding_provider.query(query)
    if not result.hits:
        return (
            "I could not find a matching catalog entry in the current catalog snapshot, so I cannot verify that detail.",
            result,
        )
    hit = result.hits[0]
    labels = ", ".join(citation.label for citation in hit.citations[:3])
    return f"{hit.title}\n{hit.content}\n[source: {labels}]".strip(), result


def evaluate_catalog_benchmark(
    responses_path: str,
    *,
    model=None,
    tokenizer_path: str | None = None,
    catalog_dsn: str | None = None,
    catalog_input_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    stop_strings: list[str] | None = None,
) -> CatalogBenchmarkResult:
    rows = [json.loads(line) for line in Path(responses_path).read_text().splitlines() if line.strip()]
    orchestrator = None
    backend = None
    grounding_provider = None
    stop_markers = list(stop_strings or ["</s>", "<|assistant|>", "<|user|>", "<|system|>", "<|tool|>", "<|citation|>"])
    if rows and any("messages" in row for row in rows):
        if model is None or tokenizer_path is None or catalog_dsn is None or catalog_input_path is None:
            raise RuntimeError(
                "Catalog benchmarks with `messages` require a model, tokenizer_path, catalog_dsn, and catalog_input_path."
            )
        from grounding.ingest import ingest_catalog
        from grounding.provider import GroundingProvider
        from grounding.store import CatalogStore
        from serve.orchestrator import AssistantOrchestrator
        from serve.types import ChatMessage

        ingest_catalog(catalog_dsn, catalog_input_path)
        store = CatalogStore(catalog_dsn)
        grounding_provider = GroundingProvider(store)
        backend = _EvalModelBackend(model, tokenizer_path)
        orchestrator = AssistantOrchestrator(
            backend,
            grounding_provider=grounding_provider,
            default_max_tokens=max_new_tokens,
            default_temperature=temperature,
            default_top_p=top_p,
            default_repetition_penalty=repetition_penalty,
            default_no_repeat_ngram_size=no_repeat_ngram_size,
            default_stop_strings=stop_markers,
            decode_preset="eval",
            backend_name="internal",
        )
        chat_message_cls = ChatMessage
    else:
        chat_message_cls = None

    exactness = 0.0
    citation_rate = 0
    abstention_rate = 0
    lane_totals: dict[str, dict[str, float]] = {}
    retrieval_audit = {
        "queries": len(rows),
        "router_triggers": 0,
        "present_queries": 0,
        "missing_queries": 0,
        "retrieval_hits": 0,
        "retrieval_misses": 0,
        "retrieval_false_negatives": 0,
        "missing_with_hits": 0,
        "true_missing_answers": 0,
        "model_hallucinations_after_no_hit": 0,
    }
    for row in rows:
        response = row.get("response")
        row_messages = row.get("messages")
        if row_messages and orchestrator is not None and grounding_provider is not None and backend is not None:
            user_query = _latest_user_message(row_messages)
            parsed_messages = [chat_message_cls(**message) for message in row_messages] if chat_message_cls else []
            _, should_ground, _ = orchestrator._route_decision(parsed_messages)
            retrieval_audit["router_triggers"] += int(should_ground)
            oracle_text, retrieval_result = _oracle_response(user_query, grounding_provider)
            if retrieval_result.hits:
                retrieval_audit["retrieval_hits"] += 1
            else:
                retrieval_audit["retrieval_misses"] += 1
            if row.get("expects_abstention", False):
                retrieval_audit["missing_queries"] += 1
                if retrieval_result.hits:
                    retrieval_audit["missing_with_hits"] += 1
                else:
                    retrieval_audit["true_missing_answers"] += 1
            else:
                retrieval_audit["present_queries"] += 1
                if not retrieval_result.hits:
                    retrieval_audit["retrieval_false_negatives"] += 1

            prompt = format_chat(row_messages, add_generation_prompt=True)
            model_only_text = backend.generate(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_strings=stop_markers,
            )
            pipeline_reply = orchestrator.respond(
                row_messages,
                tools=bool(row.get("tools", True)),
                citations=bool(row.get("citations", True)),
            )
            lane_responses = {
                "model_only": model_only_text,
                "pipeline_grounded": pipeline_reply.text,
                "retrieval_oracle": oracle_text,
            }
            for lane_name, lane_response in lane_responses.items():
                exact, has_citation, abstained = score_catalog_response(row, lane_response)
                metrics = lane_totals.setdefault(
                    lane_name,
                    {"exactness": 0.0, "citation_rate": 0.0, "abstention_rate": 0.0, "examples": 0.0},
                )
                metrics["exactness"] += exact
                metrics["citation_rate"] += float(has_citation)
                metrics["abstention_rate"] += float(abstained)
                metrics["examples"] += 1.0
            if row.get("expects_abstention", False) and not retrieval_result.hits:
                _, _, model_only_abstained = score_catalog_response(row, model_only_text)
                if not model_only_abstained:
                    retrieval_audit["model_hallucinations_after_no_hit"] += 1
            response = pipeline_reply.text
        if not isinstance(response, str):
            if orchestrator is None:
                raise RuntimeError(
                    f"Catalog benchmark row in {responses_path} is missing a `response` and no model-backed evaluator was configured."
                )
            reply = orchestrator.respond(
                row["messages"],
                tools=bool(row.get("tools", True)),
                citations=bool(row.get("citations", True)),
            )
            response = reply.text
        exact, has_citation, abstained = score_catalog_response(row, response)
        exactness += exact
        citation_rate += int(has_citation)
        abstention_rate += int(abstained)
    total = max(len(rows), 1)
    attribution_lanes = {}
    for lane_name, metrics in lane_totals.items():
        lane_examples = max(int(metrics["examples"]), 1)
        attribution_lanes[lane_name] = {
            "examples": int(metrics["examples"]),
            "exactness": float(metrics["exactness"] / lane_examples),
            "citation_rate": float(metrics["citation_rate"] / lane_examples),
            "abstention_rate": float(metrics["abstention_rate"] / lane_examples),
        }
    return CatalogBenchmarkResult(
        path=responses_path,
        examples=len(rows),
        exactness=exactness / total,
        citation_rate=citation_rate / total,
        abstention_rate=abstention_rate / total,
        attribution_lanes=attribution_lanes,
        retrieval_audit=retrieval_audit,
    )
