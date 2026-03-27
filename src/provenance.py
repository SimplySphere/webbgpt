from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    return _sha256_bytes(target.read_bytes())


def directory_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(item for item in target.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(target)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _count_examples(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def benchmark_manifest(paths: list[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        entries.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "examples": _count_examples(path),
            }
        )
    return {
        "entries": entries,
        "version": _json_hash(entries),
    }


def scorer_manifest() -> dict[str, Any]:
    assistant_path = Path("src/eval/assistant.py")
    catalog_path = Path("src/eval/catalog.py")
    webb_path = Path("src/eval/webb.py")
    payload = {
        "assistant": {"path": str(assistant_path), "sha256": file_sha256(assistant_path)},
        "catalog": {"path": str(catalog_path), "sha256": file_sha256(catalog_path)},
        "webb": {"path": str(webb_path), "sha256": file_sha256(webb_path)},
    }
    payload["version"] = _json_hash(payload)
    return payload


def release_gate_manifest(config_payload: dict[str, Any]) -> dict[str, Any]:
    return {"config": config_payload, "version": _json_hash(config_payload)}


def grounding_snapshot_manifest(
    grounding_dsn: str,
    *,
    snapshot_id: str | None = None,
    seed_url_pack: str | None = None,
    offline_seed_url_pack: str | None = None,
    handbook_url: str | None = None,
    source_policy_path: str | None = None,
    catalog_input_path: str | None = None,
) -> dict[str, Any]:
    seed_path = Path(seed_url_pack) if seed_url_pack else None
    offline_seed_path = Path(offline_seed_url_pack) if offline_seed_url_pack else None
    handbook_path = Path(handbook_url) if handbook_url else None
    source_policy = Path(source_policy_path) if source_policy_path else None
    catalog_path = Path(catalog_input_path) if catalog_input_path else None
    sqlite_file = None
    sqlite_hash = None
    if grounding_dsn.startswith("sqlite:///"):
        sqlite_file = grounding_dsn.removeprefix("sqlite:///")
        sqlite_hash = file_sha256(sqlite_file)
    family_metadata = None
    if snapshot_id:
        try:
            from grounding.store import WebbKnowledgeStore

            store = WebbKnowledgeStore(grounding_dsn)
            family_metadata = store.snapshot_family_metadata(snapshot_id)
        except Exception:
            family_metadata = None
    payload = {
        "grounding_dsn": grounding_dsn,
        "snapshot_id": snapshot_id,
        "seed_url_pack": str(seed_path) if seed_path is not None else None,
        "seed_url_pack_sha256": file_sha256(seed_path) if seed_path is not None else None,
        "offline_seed_url_pack": str(offline_seed_path) if offline_seed_path is not None else None,
        "offline_seed_url_pack_sha256": file_sha256(offline_seed_path) if offline_seed_path is not None else None,
        "source_policy_path": str(source_policy) if source_policy is not None else None,
        "source_policy_sha256": file_sha256(source_policy) if source_policy is not None else None,
        "handbook_url": handbook_url,
        "handbook_sha256": file_sha256(handbook_path) if handbook_path is not None and handbook_path.exists() else None,
        "catalog_input_path": str(catalog_path) if catalog_path is not None else None,
        "catalog_input_sha256": file_sha256(catalog_path) if catalog_path is not None else None,
        "sqlite_path": sqlite_file,
        "sqlite_sha256": sqlite_hash,
        "family_metadata": family_metadata,
    }
    payload["snapshot_id"] = snapshot_id or _json_hash(payload)
    return payload


def catalog_snapshot_manifest(catalog_dsn: str, catalog_input_path: str | None = None) -> dict[str, Any]:
    payload = grounding_snapshot_manifest(
        catalog_dsn,
        snapshot_id=None,
        catalog_input_path=catalog_input_path,
    )
    payload["catalog_dsn"] = catalog_dsn
    return payload


def checkpoint_manifest(checkpoint_path: str) -> dict[str, Any]:
    target = Path(checkpoint_path)
    payload = {
        "path": str(target),
        "checkpoint_sha256": file_sha256(target / "checkpoint.pt"),
        "directory_sha256": directory_sha256(target),
    }
    payload["artifact_id"] = _json_hash(payload)
    return payload


def tokenizer_manifest(tokenizer_path: str) -> dict[str, Any]:
    target = Path(tokenizer_path)
    if target.is_dir():
        payload = {
            "path": str(target),
            "sha256": directory_sha256(target),
            "tokenizer_model_sha256": file_sha256(target / "tokenizer.model"),
            "tokenizer_vocab_sha256": file_sha256(target / "tokenizer.vocab"),
            "tokenizer_config_sha256": file_sha256(target / "tokenizer_config.json"),
            "special_tokens_map_sha256": file_sha256(target / "special_tokens_map.json"),
        }
    else:
        vocab_path = target.with_suffix(".vocab")
        meta_path = target.with_suffix(".tokenizer.json")
        payload = {
            "path": str(target),
            "sha256": file_sha256(target),
            "vocab_path": str(vocab_path) if vocab_path.exists() else None,
            "vocab_sha256": file_sha256(vocab_path) if vocab_path.exists() else None,
            "metadata_path": str(meta_path) if meta_path.exists() else None,
            "metadata_sha256": file_sha256(meta_path) if meta_path.exists() else None,
        }
    payload["artifact_id"] = _json_hash(payload)
    return payload


def export_manifest(export_path: str | None) -> dict[str, Any] | None:
    if not export_path:
        return None
    target = Path(export_path)
    if not target.exists():
        return None
    payload = {
        "path": str(target),
        "directory_sha256": directory_sha256(target),
        "config_sha256": file_sha256(target / "config.json"),
        "generation_config_sha256": file_sha256(target / "generation_config.json"),
        "tokenizer_config_sha256": file_sha256(target / "tokenizer_config.json"),
    }
    payload["artifact_id"] = _json_hash(payload)
    return payload


def reliability_payload(examples: int, *, warn_if_below: int = 8) -> dict[str, Any]:
    swing = 0.0 if examples <= 0 else 1.0 / float(examples)
    warning = None
    if examples < warn_if_below:
        warning = (
            f"Metric slice has only {examples} example(s); one example moves the rate by {swing:.3f}."
        )
    return {
        "examples": examples,
        "per_example_swing": swing,
        "warning_threshold": warn_if_below,
        "warning": warning,
    }
