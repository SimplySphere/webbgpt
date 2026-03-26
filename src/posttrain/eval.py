from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from tokenizer import SentencePieceTokenizer, format_chat


POSTTRAIN_REGRESSION_PATH = "data/eval/posttrain_regression.jsonl"
SPECIAL_TOKEN_PIECES = {
    "<s>",
    "</s>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
}


def _require_torch():
    import torch

    return torch


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_messages(
    messages: list[dict[str, str]],
    *,
    include_assistant: bool,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        role = str(raw_message.get("role", "")).strip()
        if not include_assistant and role == "assistant":
            continue
        content = raw_message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        normalized.append({"role": role, "content": _normalize_text(content)})
    return normalized


def prompt_signature_text(messages: list[dict[str, str]]) -> str:
    return json.dumps(_normalize_messages(messages, include_assistant=False), sort_keys=True, separators=(",", ":"))


def prompt_signature_hash(messages: list[dict[str, str]]) -> str:
    encoded = prompt_signature_text(messages).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_posttrain_regression_records(
    path: str | Path = POSTTRAIN_REGRESSION_PATH,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    regression_path = Path(path)
    if not regression_path.exists():
        return records
    for line in regression_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        records.append(
            {
                "source_path": str(regression_path),
                "messages": messages,
                "tags": list(row.get("tags", [])),
                "prompt_signature_text": prompt_signature_text(messages),
                "prompt_signature_hash": prompt_signature_hash(messages),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def collect_prompt_signatures(examples: list[Any]) -> tuple[set[str], set[str]]:
    signature_texts: set[str] = set()
    signature_hashes: set[str] = set()
    for example in examples:
        if hasattr(example, "messages"):
            messages = getattr(example, "messages")
        elif hasattr(example, "prompt"):
            messages = getattr(example, "prompt")
        else:
            continue
        text = prompt_signature_text(messages)
        signature_texts.add(text)
        signature_hashes.add(prompt_signature_hash(messages))
    return signature_texts, signature_hashes


def ensure_no_regression_prompt_overlap(
    *,
    stage_name: str,
    train_examples: list[Any],
    validation_examples: list[Any],
    regression_path: str | Path = POSTTRAIN_REGRESSION_PATH,
) -> None:
    regression_records = load_posttrain_regression_records(regression_path)
    if not regression_records:
        return
    train_texts, train_hashes = collect_prompt_signatures(train_examples)
    validation_texts, validation_hashes = collect_prompt_signatures(validation_examples)
    for record in regression_records:
        prompt_text = record["prompt_signature_text"]
        prompt_hash = record["prompt_signature_hash"]
        if prompt_text in train_texts or prompt_hash in train_hashes:
            raise RuntimeError(
                f"WebbGPT: {stage_name} regression prompt suite overlaps the training data. "
                f"Remove or rewrite the overlapping prompt from {record['source_path']} before training."
            )
        if prompt_text in validation_texts or prompt_hash in validation_hashes:
            raise RuntimeError(
                f"WebbGPT: {stage_name} regression prompt suite overlaps the validation data. "
                f"Remove or rewrite the overlapping prompt from {record['source_path']} before training."
            )


def append_eval_history(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_selection_metadata(best_dir: str | Path, payload: dict[str, Any]) -> None:
    target_dir = Path(best_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "selection.json").write_text(json.dumps(payload, indent=2))


def update_topk_candidates(
    output_dir: str | Path,
    *,
    candidate_path: str | Path,
    candidate_payload: dict[str, Any],
    metric_key: str,
    limit: int,
    lower_is_better: bool = True,
) -> list[dict[str, Any]]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / "topk.json"
    current: list[dict[str, Any]] = []
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text())
    current = [entry for entry in current if entry.get("path") != str(candidate_path)]
    current.append(
        {
            "path": str(candidate_path),
            "metric_key": metric_key,
            "metric_value": candidate_payload.get("selection_value"),
            "step": candidate_payload.get("step"),
            "selection_metric": candidate_payload.get("selection_metric"),
            "metrics": candidate_payload.get("metrics"),
        }
    )
    current.sort(
        key=lambda entry: (
            float(entry.get("metric_value", float("inf"))) if lower_is_better else -float(entry.get("metric_value", float("-inf"))),
            int(entry.get("step", 0)),
        )
    )
    kept = current[: max(limit, 0)]
    kept_paths = {entry["path"] for entry in kept}
    for entry in current[max(limit, 0) :]:
        stale = Path(entry["path"])
        if stale.exists() and stale.is_dir() and str(stale) not in kept_paths:
            shutil.rmtree(stale, ignore_errors=True)
    metadata_path.write_text(json.dumps(kept, indent=2))
    return kept


def _clean_generated_response(tokenizer: SentencePieceTokenizer, token_ids: list[int]) -> tuple[str, str]:
    eos_token_id = tokenizer.token_to_id("</s>")
    raw_response = tokenizer.decode(token_ids).strip()
    clean_token_ids: list[int] = []
    for token_id in token_ids:
        if token_id == eos_token_id:
            break
        piece = tokenizer.id_to_token(token_id)
        if piece in SPECIAL_TOKEN_PIECES or (piece.startswith("<|") and piece.endswith("|>")):
            continue
        clean_token_ids.append(token_id)
    clean_response = tokenizer.decode(clean_token_ids).replace("</s>", "").strip()
    return raw_response, clean_response


def generate_qualitative_samples(
    model,
    tokenizer_path: str,
    *,
    regression_path: str | Path = POSTTRAIN_REGRESSION_PATH,
    limit: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    stop_strings: list[str] | None = None,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    prompt_records = load_posttrain_regression_records(regression_path, limit=limit)
    if not prompt_records:
        return []

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    stop_token_ids = [tokenizer.token_to_id("</s>")]
    if stop_strings:
        for value in stop_strings:
            try:
                token_id = tokenizer.token_to_id(value)
            except Exception:
                continue
            if token_id >= 0 and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
    generation_model = model.module if hasattr(model, "module") and hasattr(model.module, "generate") else model
    device = next(generation_model.parameters()).device
    previous_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        samples: list[dict[str, Any]] = []
        for record in prompt_records:
            messages = record["messages"]
            rendered = format_chat(messages, add_generation_prompt=True)
            input_ids = torch.tensor(
                [tokenizer.encode(rendered, add_bos=True, add_eos=False)],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            generated = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_token_ids=stop_token_ids,
            )
            new_tokens = generated[0, input_ids.size(1) :].tolist()
            raw_response, clean_response = _clean_generated_response(tokenizer, new_tokens)
            prompt = next(
                (message["content"] for message in reversed(messages) if message.get("role") == "user"),
                messages[-1]["content"],
            )
            samples.append(
                {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "clean_response": clean_response,
                    "tags": record["tags"],
                    "source_path": record["source_path"],
                }
            )
        return samples
    finally:
        model.train(previous_training)
