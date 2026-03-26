from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from config import GroundingConfig, ServeConfig
from grounding.ingest import webb_sync
from grounding.provider import WebbGroundingProvider
from grounding.store import WebbKnowledgeStore
from provenance import export_manifest, grounding_snapshot_manifest, tokenizer_manifest
from repro import seed_everything
from serve.backends.transformers_backend import TransformersChatBackend
from serve.backends.vllm_backend import VLLMChatBackend
from serve.orchestrator import AssistantOrchestrator
from serve.playground import render_playground_html


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageModel]
    tools: bool = True
    citations: bool = True
    safe_decode: bool = False


class ChatResponse(BaseModel):
    text: str
    used_tools: bool
    citations: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def _append_transcript(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_repro_capsule(provenance: dict, response_metadata: dict) -> dict:
    checkpoint = provenance.get("checkpoint") or {}
    tokenizer = provenance.get("tokenizer") or {}
    catalog_snapshot = provenance.get("catalog_snapshot") or {}
    generation = response_metadata.get("generation") or {}
    return {
        "checkpoint_artifact_id": checkpoint.get("artifact_id"),
        "tokenizer_artifact_id": tokenizer.get("artifact_id"),
        "snapshot_id": catalog_snapshot.get("snapshot_id"),
        "decode_preset": generation.get("decode_preset") or provenance.get("decode", {}).get("preset"),
        "backend": generation.get("backend"),
        "seed_bundle": provenance.get("seed_bundle"),
    }


def build_app(config: ServeConfig) -> FastAPI:
    seed_bundle = seed_everything(config.seed)
    try:
        backend = VLLMChatBackend(config)
    except RuntimeError:
        backend = TransformersChatBackend(config)
    grounding_config = (
        GroundingConfig.from_dict(config.grounding.to_dict()) if config.enable_grounding and config.grounding else None
    )
    if config.enable_grounding and grounding_config is None:
        grounding_config = GroundingConfig()
    grounding_store = WebbKnowledgeStore(grounding_config.dsn) if grounding_config is not None else None
    if grounding_store is not None:
        grounding_store.create_schema()
    resolved_snapshot_id = None
    sync_status: dict[str, object] | None = None
    needs_sync = False
    if grounding_config is not None and grounding_store is not None:
        needs_sync = grounding_config.sync_on_start or grounding_store.get_snapshot(grounding_config.snapshot_id) is None
    if grounding_config and needs_sync:
        sync_mode = "sync_on_start" if grounding_config.sync_on_start else "bootstrap_on_missing_snapshot"
        try:
            sync_result = webb_sync(
                grounding_config.dsn,
                seed_url_pack=grounding_config.seed_url_pack,
                source_policy_path=grounding_config.source_policy_path,
                handbook_url=grounding_config.handbook_url,
                allow_ocr_fallback=grounding_config.allow_ocr_fallback,
                label=f"{config.model_name}-serve-sync",
                families=grounding_config.sync_families or None,
            )
            resolved_snapshot_id = sync_result.get("snapshot_id")
            sync_status = {
                "mode": sync_mode,
                "status": "completed",
                "snapshot_id": resolved_snapshot_id,
            }
        except Exception as exc:
            fallback_store = grounding_store
            fallback_snapshot = fallback_store.latest_completed_snapshot()
            if fallback_snapshot is None:
                raise RuntimeError(
                    "Webb grounding sync failed and no completed trusted grounding snapshot was available."
                ) from exc
            resolved_snapshot_id = fallback_snapshot.id
            sync_status = {
                "mode": sync_mode,
                "status": "failed_using_latest_completed_snapshot",
                "snapshot_id": resolved_snapshot_id,
                "error": str(exc),
            }
    if grounding_config and grounding_store is not None:
        if resolved_snapshot_id is None:
            resolved_snapshot_id = grounding_store.resolve_snapshot_id(grounding_config.snapshot_id)
        catalog_snapshot = grounding_snapshot_manifest(
            grounding_config.dsn,
            snapshot_id=resolved_snapshot_id,
            seed_url_pack=grounding_config.seed_url_pack,
            handbook_url=grounding_config.handbook_url,
            source_policy_path=grounding_config.source_policy_path,
            catalog_input_path=grounding_config.legacy_catalog_input_path,
        )
    else:
        catalog_snapshot = {}
    provenance = {
        "checkpoint": export_manifest(config.checkpoint_path) or {"path": config.checkpoint_path},
        "tokenizer": tokenizer_manifest(config.tokenizer_path),
        "catalog_snapshot": catalog_snapshot,
        "grounding_snapshot": catalog_snapshot,
        "decode": {
            "preset": config.decode_preset,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "stop_strings": config.stop_strings,
        },
        "seed_bundle": seed_bundle,
    }
    if sync_status is not None:
        provenance["sync_on_start"] = sync_status
    grounding_provider = None
    if config.enable_grounding and grounding_config and grounding_store is not None:
        grounding_provider = WebbGroundingProvider(
            grounding_store,
            snapshot_id=resolved_snapshot_id,
            route_fanout_limit=grounding_config.route_fanout_limit,
            planner_beta_enabled=grounding_config.planner_beta_enabled,
        )
    orchestrator = AssistantOrchestrator(
        backend,
        grounding_provider=grounding_provider,
        default_max_tokens=config.max_new_tokens,
        default_temperature=config.temperature,
        default_top_p=config.top_p,
        default_repetition_penalty=config.repetition_penalty,
        default_no_repeat_ngram_size=config.no_repeat_ngram_size,
        default_stop_strings=config.stop_strings,
        decode_preset=config.decode_preset,
        backend_name=getattr(backend, "backend_name", backend.__class__.__name__),
        catalog_snapshot=catalog_snapshot,
    )

    app = FastAPI(title="WebbGPT")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return render_playground_html(config)

    @app.get("/status")
    async def status() -> dict[str, object]:
        return {
            "name": "WebbGPT",
            "status": "ok",
            "endpoints": ["/", "/status", "/healthz", "/v1/chat/completions", "/docs"],
            "provenance": provenance,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> ChatResponse:
        response = orchestrator.respond(
            [message.model_dump() for message in request.messages],
            tools=request.tools,
            citations=request.citations,
            safe_decode=request.safe_decode,
        )
        response.metadata.setdefault("provenance", provenance)
        response.metadata.setdefault("repro_capsule", _build_repro_capsule(provenance, response.metadata))
        if config.transcript_path:
            _append_transcript(
                config.transcript_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request": request.model_dump(),
                    "response": {
                        "text": response.text,
                        "used_tools": response.used_tools,
                        "citations": [citation.__dict__ for citation in response.citations],
                        "metadata": response.metadata,
                    },
                },
            )
        return ChatResponse(
            text=response.text,
            used_tools=response.used_tools,
            citations=[citation.__dict__ for citation in response.citations],
            metadata=response.metadata,
        )

    return app


def run_server(config: ServeConfig) -> None:
    app = build_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
