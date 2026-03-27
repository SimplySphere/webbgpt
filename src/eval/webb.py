from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generation import resolve_stop_token_ids, strip_stop_strings
from grounding.ingest import webb_sync
from grounding.provider import WebbGroundingProvider
from grounding.store import WebbKnowledgeStore
from serve.orchestrator import AssistantOrchestrator
from serve.types import ChatMessage
from tokenizer import SentencePieceTokenizer, format_chat


def _require_torch():
    import torch

    return torch


@dataclass(slots=True)
class WebbBenchmarkResult:
    path: str
    domain: str
    examples: int
    exactness: float
    citation_rate: float
    abstention_rate: float
    attribution_lanes: dict[str, dict[str, float]] = field(default_factory=dict)
    retrieval_audit: dict[str, Any] = field(default_factory=dict)
    route_audit: dict[str, Any] = field(default_factory=dict)


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


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    return next(
        (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
        messages[-1].get("content", "") if messages else "",
    )


def _did_abstain(response_lower: str) -> bool:
    markers = (
        "i don't know",
        "i do not know",
        "not enough information",
        "insufficient information",
        "i cannot confirm",
        "i can't confirm",
        "could not find",
        "couldn't find",
        "cannot verify",
        "don't want to guess",
        "not listed",
    )
    return any(marker in response_lower for marker in markers)


def score_webb_response(record: dict[str, Any], response: str) -> tuple[float, bool, bool]:
    response_lower = response.lower()
    has_citation = "[source:" in response_lower or "<|citation|>" in response_lower
    abstained = _did_abstain(response_lower)
    expects_abstention = bool(record.get("expects_abstention", False))
    expected_codes = [value.lower() for value in record.get("expected_course_codes", [])]
    expected_substrings = [value.lower() for value in record.get("expected_substrings", [])]
    expected_any = [value.lower() for value in record.get("expected_any_substrings", [])]
    exact = True
    if expects_abstention:
        exact = abstained
    else:
        if expected_codes:
            exact = exact and all(code in response_lower for code in expected_codes)
        if expected_substrings:
            exact = exact and all(fragment in response_lower for fragment in expected_substrings)
        if expected_any:
            exact = exact and any(fragment in response_lower for fragment in expected_any)
        if not expected_codes and not expected_substrings and not expected_any:
            exact = bool(response.strip())
    return float(exact), has_citation, abstained


def _oracle_response(query: str, provider: WebbGroundingProvider, route: str) -> tuple[str, Any]:
    result = provider.query(query, route=route)
    if not result.hits:
        return provider.no_hit_message(route, query), result
    return provider.render_context(result), result


def _expected_routes(record: dict[str, Any]) -> list[str]:
    routes = record.get("expected_routes")
    if isinstance(routes, list):
        normalized = [WebbGroundingProvider.normalize_route(str(route)) for route in routes if str(route).strip()]
        if normalized:
            return normalized
    explicit = str(record.get("expected_route", "")).strip()
    if explicit:
        return [WebbGroundingProvider.normalize_route(explicit)]
    domain = str(record.get("domain", "")).lower()
    if "handbook" in domain:
        return ["handbook_policy"]
    if "faculty" in domain:
        return ["faculty"]
    if "admission" in domain:
        return ["admissions_general"]
    if "student_life" in domain:
        return ["student_life"]
    if "mission_values" in domain or "college_guidance" in domain:
        return ["admissions_general"]
    if "museum" in domain:
        return ["museum_programs"]
    if "athletics" in domain:
        return ["athletics"]
    if "planner" in domain:
        return ["planner_advice"]
    if "course" in domain:
        return ["course_catalog"]
    return ["chat"]


def evaluate_webb_benchmark(
    responses_path: str,
    *,
    model=None,
    tokenizer_path: str | None = None,
    grounding_dsn: str,
    seed_url_pack: str,
    offline_seed_url_pack: str | None = None,
    handbook_url: str | None = None,
    source_policy_path: str | None = None,
    snapshot_id: str = "latest",
    sync_on_start: bool = False,
    allow_ocr_fallback: bool = False,
    route_fanout_limit: int = 2,
    planner_beta_enabled: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    stop_strings: list[str] | None = None,
) -> WebbBenchmarkResult:
    rows = [json.loads(line) for line in Path(responses_path).read_text().splitlines() if line.strip()]
    stop_markers = list(stop_strings or ["</s>", "<|assistant|>", "<|user|>", "<|system|>", "<|tool|>", "<|citation|>"])
    store = WebbKnowledgeStore(grounding_dsn)
    store.create_schema()
    resolved_snapshot_id = None
    if sync_on_start or store.get_snapshot(snapshot_id) is None:
        try:
            sync_result = webb_sync(
                grounding_dsn,
                seed_url_pack=seed_url_pack,
                offline_seed_url_pack=offline_seed_url_pack,
                source_policy_path=source_policy_path,
                handbook_url=handbook_url,
                allow_ocr_fallback=allow_ocr_fallback,
                label="webb-eval-sync",
            )
            resolved_snapshot_id = str(sync_result.get("snapshot_id") or "")
        except Exception as exc:
            fallback_snapshot = store.latest_completed_snapshot()
            if fallback_snapshot is None:
                raise RuntimeError(
                    "Webb evaluation sync failed and no completed trusted grounding snapshot was available."
                ) from exc
            resolved_snapshot_id = fallback_snapshot.id
    if not resolved_snapshot_id:
        resolved_snapshot_id = store.resolve_snapshot_id(snapshot_id)
    provider = WebbGroundingProvider(
        store,
        snapshot_id=resolved_snapshot_id,
        route_fanout_limit=route_fanout_limit,
        planner_beta_enabled=planner_beta_enabled,
    )
    backend = None
    orchestrator = None
    if rows and any("messages" in row for row in rows):
        if model is None or tokenizer_path is None:
            raise RuntimeError("Webb grounded benchmarks with messages require a model and tokenizer_path.")
        backend = _EvalModelBackend(model, tokenizer_path)
        orchestrator = AssistantOrchestrator(
            backend,
            grounding_provider=provider,
            default_max_tokens=max_new_tokens,
            default_temperature=temperature,
            default_top_p=top_p,
            default_repetition_penalty=repetition_penalty,
            default_no_repeat_ngram_size=no_repeat_ngram_size,
            default_stop_strings=stop_markers,
            decode_preset="eval",
            backend_name="internal",
            catalog_snapshot={"snapshot_id": resolved_snapshot_id, "seed_url_pack": seed_url_pack},
        )

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
    route_audit = {
        "queries": len(rows),
        "expected_grounded_queries": 0,
        "route_true_positives": 0,
        "route_false_negatives": 0,
        "mismatched_routes": 0,
    }
    domain = str(rows[0].get("domain") or Path(responses_path).stem) if rows else Path(responses_path).stem
    for row in rows:
        response = row.get("response")
        row_messages = row.get("messages")
        if row_messages and orchestrator is not None and backend is not None:
            user_query = _latest_user_message(row_messages)
            parsed_messages = [ChatMessage(**message) for message in row_messages]
            route_decision = provider.route_messages(parsed_messages)
            expected_routes = _expected_routes(row)
            candidate_routes = list((route_decision.metadata or {}).get("fanout_routes") or [route_decision.route])
            retrieval_audit["router_triggers"] += int(route_decision.grounded or bool(candidate_routes))
            if expected_routes != ["chat"]:
                route_audit["expected_grounded_queries"] += 1
                if any(route in candidate_routes for route in expected_routes):
                    route_audit["route_true_positives"] += 1
                else:
                    route_audit["route_false_negatives"] += 1
            if route_decision.route not in expected_routes:
                route_audit["mismatched_routes"] += 1
            oracle_blocks: list[str] = []
            retrieval_hits = []
            last_result = None
            for candidate_route in candidate_routes:
                oracle_text_part, retrieval_result = _oracle_response(user_query, provider, candidate_route)
                last_result = retrieval_result
                if retrieval_result.hits:
                    retrieval_hits.extend(retrieval_result.hits)
                    oracle_blocks.append(f"[{candidate_route}]\n{oracle_text_part}")
            if retrieval_hits:
                retrieval_audit["retrieval_hits"] += 1
            else:
                retrieval_audit["retrieval_misses"] += 1
            if row.get("expects_abstention", False):
                retrieval_audit["missing_queries"] += 1
                if retrieval_hits:
                    retrieval_audit["missing_with_hits"] += 1
                else:
                    retrieval_audit["true_missing_answers"] += 1
            else:
                retrieval_audit["present_queries"] += 1
                if not retrieval_hits:
                    retrieval_audit["retrieval_false_negatives"] += 1
            oracle_text = "\n\n".join(oracle_blocks) if oracle_blocks else provider.no_hit_message(route_decision.route, user_query)

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
                tools=True,
                citations=True,
            )
            lane_responses = {
                "model_only": model_only_text,
                "pipeline_grounded": pipeline_reply.text,
                "retrieval_oracle": oracle_text,
            }
            for lane_name, lane_response in lane_responses.items():
                exact, has_citation, abstained = score_webb_response(row, lane_response)
                metrics = lane_totals.setdefault(
                    lane_name,
                    {"exactness": 0.0, "citation_rate": 0.0, "abstention_rate": 0.0, "examples": 0.0},
                )
                metrics["exactness"] += exact
                metrics["citation_rate"] += float(has_citation)
                metrics["abstention_rate"] += float(abstained)
                metrics["examples"] += 1.0
            if row.get("expects_abstention", False) and not retrieval_hits:
                _, _, model_only_abstained = score_webb_response(row, model_only_text)
                if not model_only_abstained:
                    retrieval_audit["model_hallucinations_after_no_hit"] += 1
            response = pipeline_reply.text
        if not isinstance(response, str):
            raise RuntimeError(f"Webb benchmark row in {responses_path} is missing a response.")
        exact, has_citation, abstained = score_webb_response(row, response)
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
    if route_audit["expected_grounded_queries"] > 0:
        route_audit["route_false_negative_rate"] = (
            route_audit["route_false_negatives"] / route_audit["expected_grounded_queries"]
        )
    return WebbBenchmarkResult(
        path=responses_path,
        domain=domain,
        examples=len(rows),
        exactness=exactness / total,
        citation_rate=citation_rate / total,
        abstention_rate=abstention_rate / total,
        attribution_lanes=attribution_lanes,
        retrieval_audit=retrieval_audit,
        route_audit=route_audit,
    )
