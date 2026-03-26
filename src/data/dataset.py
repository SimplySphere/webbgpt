from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from config import DataConfig, DataSourceConfig
from data.packing import PackedSequencePacker, pack_token_sequences
from data.prepared import (
    PreparedPackedDataset,
    PreparedPreferenceDataset,
    PreparedSFTDataset,
    append_hash_chunk,
    build_input_fingerprint,
    cleanup_prepare_outputs,
    encode_preference_example,
    encode_sft_messages,
    load_prepared_manifest,
    load_buffer_rows,
    load_seen_hashes,
    prepared_resume_dir,
    prepared_resume_state_path,
    remove_resume_artifacts,
    save_buffer_rows,
    save_prepared_manifest,
    save_resume_state,
    stage_has_partial_outputs,
    validate_resume_state_files,
)
from data.preprocess import clean_document
from data.schemas import DocumentRecord, PreferenceExample, SFTExample
from tokenizer import SentencePieceTokenizer


PREPARE_DOC_SNAPSHOT_INTERVAL = 1_000
PREPARE_EXAMPLE_SNAPSHOT_INTERVAL = 100


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _require_datasets():
    try:
        from datasets import Dataset, IterableDataset, load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is required for data preparation. Install with `pip install datasets`."
        ) from exc
    return Dataset, IterableDataset, load_dataset


def _require_torch():
    try:
        import torch
        from torch.utils.data import ConcatDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for dataset objects. Install with `pip install torch`.") from exc
    return torch, ConcatDataset


def _data_files_for_source(source: DataSourceConfig):
    if source.paths:
        return source.paths
    if source.path:
        return source.path
    raise ValueError(f"Source {source.name!r} does not define a local path or shard list.")


def _apply_record_window(dataset, source: DataSourceConfig):
    dataset_cls, iterable_cls, _ = _require_datasets()
    if isinstance(dataset, iterable_cls):
        if source.skip_records > 0:
            dataset = dataset.skip(source.skip_records)
        if source.max_records is not None:
            dataset = dataset.take(source.max_records)
        return dataset

    if isinstance(dataset, dataset_cls):
        start = min(source.skip_records, len(dataset))
        stop = len(dataset) if source.max_records is None else min(start + source.max_records, len(dataset))
        return dataset.select(range(start, stop))

    return dataset


def _coerce_prompt_messages(prompt: Any) -> list[dict[str, str]] | None:
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _prompt_signature_text(messages: list[dict[str, str]]) -> str:
    return json.dumps(_normalize_messages(messages, include_assistant=False), sort_keys=True, separators=(",", ":"))


def _prompt_signature_hash(messages: list[dict[str, str]]) -> str:
    return _stable_payload_hash(_normalize_messages(messages, include_assistant=False))


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _normalize_text(str(value))
    return text or None


def _message_group_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    if source.group_field is not None:
        explicit_group = _coerce_optional_text(item.get(source.group_field))
        if explicit_group is not None:
            return explicit_group
    explicit_group = _coerce_optional_text(item.get("group_id"))
    if explicit_group is not None:
        return explicit_group
    conversation_id = _coerce_optional_text(item.get("conversation_id"))
    if conversation_id is not None:
        return conversation_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "messages": _normalize_messages(messages, include_assistant=False),
        }
    )


def _message_example_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    if source.id_field is not None:
        explicit_id = _coerce_optional_text(item.get(source.id_field))
        if explicit_id is not None:
            return explicit_id
    explicit_id = _coerce_optional_text(item.get("example_id"))
    if explicit_id is not None:
        return explicit_id
    row_id = _coerce_optional_text(item.get("id"))
    if row_id is not None:
        return row_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "messages": _normalize_messages(messages, include_assistant=True),
        }
    )


def _preference_example_id(
    source: DataSourceConfig,
    item: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    chosen: str,
    rejected: str,
) -> str:
    if source.id_field is not None:
        explicit_id = _coerce_optional_text(item.get(source.id_field))
        if explicit_id is not None:
            return explicit_id
    explicit_id = _coerce_optional_text(item.get("example_id"))
    if explicit_id is not None:
        return explicit_id
    row_id = _coerce_optional_text(item.get("id"))
    if row_id is not None:
        return row_id
    return _stable_payload_hash(
        {
            "source": source.name,
            "prompt": _normalize_messages(prompt_messages, include_assistant=False),
            "chosen": _normalize_text(chosen),
            "rejected": _normalize_text(rejected),
        }
    )


def _source_location(source: DataSourceConfig) -> str:
    return source.dataset_name or source.path or ",".join(source.paths)


def _source_with_cursor(source: DataSourceConfig, raw_records_consumed: int) -> DataSourceConfig:
    if raw_records_consumed <= 0:
        return source
    updated = DataSourceConfig.from_dict(source.to_dict())
    updated.skip_records = source.skip_records + raw_records_consumed
    if source.max_records is not None:
        updated.max_records = max(source.max_records - raw_records_consumed, 0)
    return updated


class PackedSequenceDataset:
    def __init__(self, sequences: list[list[int]], pad_token_id: int):
        self.sequences = sequences
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        attention_mask = (sequence != self.pad_token_id).long()
        labels = sequence.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": sequence, "attention_mask": attention_mask, "labels": labels}


class SFTDataset:
    def __init__(self, examples: list[SFTExample], tokenizer_path: str, sequence_length: int):
        self.examples = examples
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        example = self.examples[index]
        input_ids, labels = encode_sft_messages(example.messages, self.tokenizer, self.sequence_length)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_tensor != self.tokenizer.token_to_id("<pad>")).long()
        return {"input_ids": input_tensor, "attention_mask": attention_mask, "labels": labels_tensor}


class PreferenceDataset:
    def __init__(self, examples: list[PreferenceExample], tokenizer_path: str, sequence_length: int):
        self.examples = examples
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        self.sequence_length = sequence_length

    def _encode(self, prompt: list[dict[str, str]], answer: str) -> list[int]:
        return encode_preference_example(prompt, answer, self.tokenizer, self.sequence_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _ = _require_torch()
        example = self.examples[index]
        chosen = self._encode(example.prompt, example.chosen)
        rejected = self._encode(example.prompt, example.rejected)
        pad_id = self.tokenizer.token_to_id("<pad>")
        return {
            "chosen_input_ids": torch.tensor(chosen, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected, dtype=torch.long),
            "chosen_attention_mask": torch.tensor([token != pad_id for token in chosen], dtype=torch.long),
            "rejected_attention_mask": torch.tensor([token != pad_id for token in rejected], dtype=torch.long),
        }


class IndexedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _split_indices_by_group(
    items: list[Any],
    *,
    stage_name: str,
    seed: int,
    validation_fraction: float,
    validation_min_examples: int,
    allow_weak_validation: bool,
) -> tuple[list[int], list[int]]:
    total_examples = len(items)
    if validation_fraction <= 0:
        return list(range(total_examples)), []
    if total_examples < 2:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage needs at least 2 examples before it can reserve validation examples."
        )

    requested_val = round(total_examples * validation_fraction)
    target_val_examples = max(validation_min_examples, requested_val)
    grouped_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        group_id = getattr(item, "split_group_id", None) or getattr(item, "example_id", None) or f"{stage_name}:{index}"
        grouped_indices.setdefault(str(group_id), []).append(index)

    group_ids = list(grouped_indices)
    if len(group_ids) < 2:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage needs at least two split groups before it can create a held-out validation set."
        )

    random.Random(f"{seed}:{stage_name}:validation").shuffle(group_ids)
    validation_group_ids: list[str] = []
    validation_count = 0
    for group_id in group_ids:
        remaining_groups = len(group_ids) - len(validation_group_ids) - 1
        if validation_count >= target_val_examples and validation_group_ids:
            break
        if remaining_groups < 1:
            break
        validation_group_ids.append(group_id)
        validation_count += len(grouped_indices[group_id])

    validation_index_set = {
        index for group_id in validation_group_ids for index in grouped_indices[group_id]
    }
    train_indices = [index for index in range(total_examples) if index not in validation_index_set]
    validation_indices = [index for index in range(total_examples) if index in validation_index_set]

    if not train_indices or not validation_indices:
        raise RuntimeError(
            f"WebbGPT: {stage_name} stage could not create a non-empty grouped validation split. "
            "Provide explicit validation sources instead."
        )
    if len(validation_indices) < validation_min_examples:
        message = (
            f"WebbGPT: grouped auto-split for {stage_name} would produce only {len(validation_indices)} "
            f"validation examples, below the required minimum of {validation_min_examples}. "
            "Provide explicit validation sources instead."
        )
        if not allow_weak_validation:
            raise RuntimeError(message)
        _progress(f"{message} Continuing only because allow_weak_posttrain_validation=true.")

    _progress(
        f"WebbGPT: split {stage_name} stage into {len(train_indices)} train and {len(validation_indices)} validation examples "
        f"across {len(grouped_indices)} grouped prompts."
    )
    return train_indices, validation_indices


def split_dataset_for_validation(
    dataset,
    *,
    stage_name: str,
    seed: int,
    validation_fraction: float,
    validation_min_examples: int,
    allow_weak_validation: bool = False,
) -> tuple[object, object | None]:
    total_examples = len(dataset)
    if validation_fraction <= 0 or total_examples == 0:
        return dataset, None
    items = getattr(dataset, "examples", None)
    if items is None:
        items = [dataset[index] for index in range(total_examples)]
    train_indices, validation_indices = _split_indices_by_group(
        list(items),
        stage_name=stage_name,
        seed=seed,
        validation_fraction=validation_fraction,
        validation_min_examples=validation_min_examples,
        allow_weak_validation=allow_weak_validation,
    )
    return IndexedDataset(dataset, train_indices), IndexedDataset(dataset, validation_indices)


class DatasetBuilder:
    def __init__(self, config: DataConfig):
        self.config = config

    def _stage_sources(self, stage: str) -> list[DataSourceConfig]:
        mapping = {
            "pretrain": self.config.pretrain_sources,
            "continue": self.config.continued_pretrain_sources,
            "sft": self.config.sft_sources,
            "preference": self.config.preference_sources,
            "validation": self.config.validation_sources,
        }
        return mapping[stage]

    def _require_stage_sources(self, stage: str, sources: list[DataSourceConfig]) -> None:
        if sources:
            return
        stage_help = {
            "pretrain": "pretraining text sources",
            "continue": "continued-pretraining text sources",
            "sft": "SFT chat examples",
            "preference": "preference chosen/rejected examples",
            "validation": "validation text sources",
        }[stage]
        raise RuntimeError(
            f"No {stage_help} are configured in the current data config. "
            f"Populate the `{stage}` stage sources before running this command."
        )

    def _uses_prepared_sources(self, sources: list[DataSourceConfig]) -> bool:
        if not sources:
            return False
        prepared = [source.format == "prepared" for source in sources]
        if any(prepared) and not all(prepared):
            raise RuntimeError("Do not mix prepared-manifest sources with raw sources in the same stage.")
        return all(prepared)

    def _concat_datasets(self, datasets: list):
        if len(datasets) == 1:
            return datasets[0]
        _, concat_cls = _require_torch()
        return concat_cls(datasets)

    def _build_prepared_dataset(self, sources: list[DataSourceConfig], expected_kind: str):
        datasets = []
        for source in sources:
            manifest = load_prepared_manifest(source.path)
            kind = manifest.get("kind")
            if kind != expected_kind:
                raise RuntimeError(
                    f"Prepared source {source.path} has kind {kind!r}, expected {expected_kind!r}."
                )
            if kind == "packed_lm":
                datasets.append(PreparedPackedDataset(source.path))
            elif kind == "sft":
                datasets.append(PreparedSFTDataset(source.path))
            elif kind == "preference":
                datasets.append(PreparedPreferenceDataset(source.path))
            else:
                raise RuntimeError(f"Unsupported prepared dataset kind {kind!r}.")
        return self._concat_datasets(datasets)

    def _initial_source_progress(self, sources: list[DataSourceConfig]) -> list[dict[str, Any]]:
        return [
            {
                "name": source.name,
                "raw_records_consumed": 0,
                "accepted_records": 0,
            }
            for source in sources
        ]

    def _resolve_prepare_target(
        self,
        *,
        stage: str,
        kind: str,
        output_path: str,
        input_fingerprint: str,
        force_rebuild: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        manifest_path = Path(output_path)
        resume_state_path = prepared_resume_state_path(manifest_path)
        resume_workspace = prepared_resume_dir(manifest_path)

        if force_rebuild:
            _progress(f"WebbGPT: force rebuilding prepared stage {stage}; clearing prior outputs first.")
            cleanup_prepare_outputs(manifest_path)

        if manifest_path.exists():
            manifest = load_prepared_manifest(manifest_path)
            manifest_fingerprint = manifest.get("input_fingerprint")
            if manifest_fingerprint is None:
                raise RuntimeError(
                    f"Existing prepared manifest at {manifest_path} predates resumable metadata and is not safely reusable. "
                    "Re-run with --force-rebuild."
                )
            if manifest_fingerprint != input_fingerprint:
                raise RuntimeError(
                    f"Existing prepared manifest at {manifest_path} does not match the current {stage} inputs. "
                    "Re-run with --force-rebuild."
                )
            if manifest.get("kind") != kind:
                raise RuntimeError(
                    f"Existing prepared manifest at {manifest_path} has kind {manifest.get('kind')!r}, expected {kind!r}. "
                    "Re-run with --force-rebuild."
                )
            _progress(f"WebbGPT: reusing completed prepared stage {stage} from {manifest_path}.")
            remove_resume_artifacts(manifest_path)
            return "reuse", manifest

        if resume_state_path.exists():
            state = load_prepared_manifest(resume_state_path)
            if state.get("input_fingerprint") != input_fingerprint:
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} does not match the current {stage} inputs. "
                    "Re-run with --force-rebuild."
                )
            if state.get("kind") != kind:
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} has kind {state.get('kind')!r}, expected {kind!r}. "
                    "Re-run with --force-rebuild."
                )
            validate_resume_state_files(state)
            return "resume", state

        if stage_has_partial_outputs(manifest_path) or resume_workspace.exists():
            raise RuntimeError(
                f"Found partial prepared-data outputs for stage {stage!r} at {manifest_path.with_suffix('')} "
                "without resumable metadata. These legacy partial shards are not resumable. "
                "Re-run with --force-rebuild to discard them and rebuild safely."
            )

        return "fresh", None

    def _prepare_packed_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        pad_token_id = tokenizer.token_to_id("<pad>")
        eos_token_id = tokenizer.token_to_id("</s>")
        token_budget = None
        if stage == "pretrain":
            token_budget = self.config.pretraining_token_budget
        elif stage == "continue":
            token_budget = self.config.continued_pretraining_token_budget
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="packed_lm",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            token_budget=token_budget,
            extra={
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "packing_version": "checkpointable-v1",
            },
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="packed_lm",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        rows_buffer_path = resume_workspace / "rows-buffer.npy"

        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            rows = load_buffer_rows(state.get("rows_buffer_path"))
            packer = PackedSequencePacker(
                sequence_length=self.config.sequence_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                current=(state.get("packer_state") or {}).get("current", []),
            )
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            num_sequences = int(state.get("num_sequences", 0))
            num_tokens = int(state.get("num_tokens", 0))
            dedupe_hash_chunks = list(state.get("dedupe_hash_chunks", []))
            seen_hashes = load_seen_hashes(dedupe_hash_chunks) if any(source.deduplicate for source in sources) else set()
            _progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {num_tokens:,} packed tokens."
            )
        else:
            source_progress = self._initial_source_progress(sources)
            rows: list[list[int]] = []
            packer = PackedSequencePacker(
                sequence_length=self.config.sequence_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            shards = []
            shard_index = 0
            num_sequences = 0
            num_tokens = 0
            dedupe_hash_chunks: list[str] = []
            seen_hashes: set[str] = set()
            _progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        pending_hashes: list[str] = []
        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal pending_hashes, consumed_since_snapshot
            if pending_hashes:
                chunk_path = append_hash_chunk(
                    resume_workspace / f"dedupe-{len(dedupe_hash_chunks):05d}.txt",
                    pending_hashes,
                )
                if chunk_path is not None:
                    dedupe_hash_chunks.append(chunk_path)
                pending_hashes = []
            buffer_path = save_buffer_rows(rows_buffer_path, rows)
            if buffer_path is None and rows_buffer_path.exists():
                rows_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "1.0",
                    "stage": stage,
                    "kind": "packed_lm",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "rows_per_shard": self.config.prepared_shard_size,
                    "token_budget": token_budget,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_sequences": num_sequences,
                    "num_tokens": num_tokens,
                    "packer_state": packer.state_dict(),
                    "rows_buffer_path": buffer_path,
                    "dedupe_hash_chunks": dedupe_hash_chunks,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(*, final: bool = False) -> None:
            nonlocal rows, shard_index
            if not rows:
                return
            shard_path = shard_dir / f"shard-{shard_index:05d}.npy"
            save_buffer_rows(shard_path, rows)
            shards.append({"path": str(shard_path), "rows": len(rows)})
            message_prefix = "final shard" if final else "shard"
            _progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_sequences:,} sequences, {num_tokens:,} packed tokens so far)."
            )
            rows = []
            shard_index += 1
            snapshot_state()

        token_budget_reached = token_budget is not None and num_tokens >= token_budget
        for source_index, source in enumerate(sources):
            progress = source_progress[source_index]
            kept_records = int(progress.get("accepted_records", 0))
            if token_budget_reached:
                break
            _progress(
                f"WebbGPT: preparing {stage} source {source.name} "
                f"({source.format}) from {_source_location(source)}."
            )
            for item in self._load_source_records(
                source,
                raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
            ):
                progress["raw_records_consumed"] = int(progress.get("raw_records_consumed", 0)) + 1
                consumed_since_snapshot += 1
                text = item.get(source.text_field, "")
                if isinstance(text, str):
                    record = DocumentRecord(
                        text=text,
                        source=source.name,
                        metadata={field: item.get(field) for field in source.metadata_fields},
                    )
                    cleaned = clean_document(record, self.config, source, seen_hashes)
                    if cleaned.record is not None:
                        kept_records += 1
                        progress["accepted_records"] = kept_records
                        if cleaned.record.document_id and source.deduplicate:
                            pending_hashes.append(cleaned.record.document_id)
                        token_ids = tokenizer.encode(cleaned.record.text, add_bos=True, add_eos=True)
                        for sequence in packer.push(token_ids):
                            rows.append(sequence)
                            num_sequences += 1
                            num_tokens += sum(token != pad_token_id for token in sequence)
                            if len(rows) >= self.config.prepared_shard_size:
                                flush_completed_shard()
                            if token_budget is not None and num_tokens >= token_budget:
                                token_budget_reached = True
                                break
                        if kept_records % 1000 == 0:
                            _progress(
                                f"WebbGPT: preparing {stage} source {source.name}: "
                                f"kept {kept_records:,} documents so far."
                            )
                if consumed_since_snapshot >= PREPARE_DOC_SNAPSHOT_INTERVAL:
                    snapshot_state()
                if token_budget_reached:
                    break
            _progress(
                f"WebbGPT: preparing {stage} source {source.name}: "
                f"finished with {kept_records:,} documents kept."
            )

        if not token_budget_reached:
            for sequence in packer.finish():
                rows.append(sequence)
                num_sequences += 1
                num_tokens += sum(token != pad_token_id for token in sequence)
                if len(rows) >= self.config.prepared_shard_size:
                    flush_completed_shard()

        flush_completed_shard(final=True)
        manifest = {
            "version": "1.0",
            "stage": stage,
            "kind": "packed_lm",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "num_sequences": num_sequences,
            "num_tokens": num_tokens,
            "source_snapshots": source_snapshots,
            "shards": shards,
        }
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_sequences:,} sequences across {len(shards):,} shards, {num_tokens:,} packed tokens)."
        )
        return manifest

    def _prepare_sft_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="sft",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            extra={"label_mode": "assistant_only_v1"},
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="sft",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        input_buffer_path = resume_workspace / "input-buffer.npy"
        label_buffer_path = resume_workspace / "label-buffer.npy"

        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            input_rows = load_buffer_rows(state.get("input_buffer_path"))
            label_rows = load_buffer_rows(state.get("label_buffer_path"))
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            num_examples = int(state.get("num_examples", 0))
            num_label_tokens = int(state.get("num_label_tokens", 0))
            _progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {num_examples:,} examples."
            )
        else:
            source_progress = self._initial_source_progress(sources)
            input_rows = []
            label_rows = []
            shards = []
            shard_index = 0
            num_examples = 0
            num_label_tokens = 0
            _progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal consumed_since_snapshot
            input_buffer = save_buffer_rows(input_buffer_path, input_rows)
            label_buffer = save_buffer_rows(label_buffer_path, label_rows)
            if input_buffer is None and input_buffer_path.exists():
                input_buffer_path.unlink()
            if label_buffer is None and label_buffer_path.exists():
                label_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "1.0",
                    "stage": stage,
                    "kind": "sft",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": tokenizer.token_to_id("<pad>"),
                    "rows_per_shard": self.config.prepared_shard_size,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_examples": num_examples,
                    "num_label_tokens": num_label_tokens,
                    "input_buffer_path": input_buffer,
                    "label_buffer_path": label_buffer,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(*, final: bool = False) -> None:
            nonlocal input_rows, label_rows, shard_index
            if not input_rows:
                return
            input_path = shard_dir / f"input_ids-{shard_index:05d}.npy"
            label_path = shard_dir / f"labels-{shard_index:05d}.npy"
            save_buffer_rows(input_path, input_rows)
            save_buffer_rows(label_path, label_rows)
            shards.append(
                {
                    "input_ids_path": str(input_path),
                    "labels_path": str(label_path),
                    "rows": len(input_rows),
                }
            )
            message_prefix = "final shard" if final else "shard"
            _progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_examples:,} examples, {num_label_tokens:,} supervised tokens so far)."
            )
            input_rows = []
            label_rows = []
            shard_index += 1
            snapshot_state()

        for source_index, source in enumerate(sources):
            progress = source_progress[source_index]
            accepted_records = int(progress.get("accepted_records", 0))
            _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(
                source,
                raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
            ):
                progress["raw_records_consumed"] = int(progress.get("raw_records_consumed", 0)) + 1
                consumed_since_snapshot += 1
                messages = item.get(source.messages_field)
                if not isinstance(messages, list):
                    prompt_messages = _coerce_prompt_messages(item.get(source.prompt_field))
                    response = item.get(source.response_field)
                    if not isinstance(response, str):
                        response = item.get(source.chosen_field)
                    if prompt_messages is None or not isinstance(response, str):
                        if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                            snapshot_state()
                        continue
                    messages = [*prompt_messages, {"role": "assistant", "content": response}]
                input_ids, labels = encode_sft_messages(messages, tokenizer, self.config.sequence_length)
                input_rows.append(input_ids)
                label_rows.append(labels)
                accepted_records += 1
                progress["accepted_records"] = accepted_records
                num_examples += 1
                num_label_tokens += sum(label != -100 for label in labels)
                if len(input_rows) >= self.config.prepared_shard_size:
                    flush_completed_shard()
                if accepted_records % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {accepted_records:,} SFT examples so far."
                    )
                if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                    snapshot_state()
            _progress(
                f"WebbGPT: preparing {stage} source {source.name}: "
                f"finished with {accepted_records:,} SFT examples."
            )

        flush_completed_shard(final=True)
        manifest = {
            "version": "1.0",
            "stage": stage,
            "kind": "sft",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "num_examples": num_examples,
            "num_label_tokens": num_label_tokens,
            "source_snapshots": source_snapshots,
            "shards": shards,
        }
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_examples:,} examples across {len(shards):,} shards, {num_label_tokens:,} supervised tokens)."
        )
        return manifest

    def _prepare_preference_stage(
        self,
        stage: str,
        sources: list[DataSourceConfig],
        output_path: str,
        *,
        force_rebuild: bool,
    ) -> dict[str, Any]:
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        source_snapshots = [source.to_dict() for source in sources]
        input_fingerprint = build_input_fingerprint(
            stage=stage,
            kind="preference",
            tokenizer_path=self.config.tokenizer_path,
            sequence_length=self.config.sequence_length,
            rows_per_shard=self.config.prepared_shard_size,
            source_snapshots=source_snapshots,
            extra={"preference_mode": "chosen_rejected_v1"},
        )
        action, payload = self._resolve_prepare_target(
            stage=stage,
            kind="preference",
            output_path=output_path,
            input_fingerprint=input_fingerprint,
            force_rebuild=force_rebuild,
        )
        if action == "reuse":
            return payload or {}

        manifest_path = Path(output_path)
        shard_dir = manifest_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        resume_workspace = prepared_resume_dir(manifest_path)
        resume_workspace.mkdir(parents=True, exist_ok=True)
        resume_state_path = prepared_resume_state_path(manifest_path)
        chosen_buffer_path = resume_workspace / "chosen-buffer.npy"
        rejected_buffer_path = resume_workspace / "rejected-buffer.npy"

        if action == "resume":
            state = payload or {}
            source_progress = list(state.get("source_progress", []))
            if len(source_progress) != len(sources):
                raise RuntimeError(
                    f"Prepared-data resume state at {resume_state_path} no longer matches the configured source list. "
                    "Re-run with --force-rebuild."
                )
            chosen_rows = load_buffer_rows(state.get("chosen_buffer_path"))
            rejected_rows = load_buffer_rows(state.get("rejected_buffer_path"))
            shards = list(state.get("shards", []))
            shard_index = int(state.get("next_shard_index", len(shards)))
            num_examples = int(state.get("num_examples", 0))
            _progress(
                f"WebbGPT: resuming prepared stage {stage} "
                f"from {len(shards):,} shard(s) and {num_examples:,} preference examples."
            )
        else:
            source_progress = self._initial_source_progress(sources)
            chosen_rows = []
            rejected_rows = []
            shards = []
            shard_index = 0
            num_examples = 0
            _progress(f"WebbGPT: starting fresh prepared stage {stage}.")

        consumed_since_snapshot = 0

        def snapshot_state() -> None:
            nonlocal consumed_since_snapshot
            chosen_buffer = save_buffer_rows(chosen_buffer_path, chosen_rows)
            rejected_buffer = save_buffer_rows(rejected_buffer_path, rejected_rows)
            if chosen_buffer is None and chosen_buffer_path.exists():
                chosen_buffer_path.unlink()
            if rejected_buffer is None and rejected_buffer_path.exists():
                rejected_buffer_path.unlink()
            save_resume_state(
                resume_state_path,
                {
                    "version": "1.0",
                    "stage": stage,
                    "kind": "preference",
                    "input_fingerprint": input_fingerprint,
                    "tokenizer_path": self.config.tokenizer_path,
                    "sequence_length": self.config.sequence_length,
                    "pad_token_id": tokenizer.token_to_id("<pad>"),
                    "rows_per_shard": self.config.prepared_shard_size,
                    "source_snapshots": source_snapshots,
                    "source_progress": source_progress,
                    "shards": shards,
                    "next_shard_index": shard_index,
                    "num_examples": num_examples,
                    "chosen_buffer_path": chosen_buffer,
                    "rejected_buffer_path": rejected_buffer,
                },
            )
            consumed_since_snapshot = 0

        def flush_completed_shard(*, final: bool = False) -> None:
            nonlocal chosen_rows, rejected_rows, shard_index
            if not chosen_rows:
                return
            chosen_path = shard_dir / f"chosen_input_ids-{shard_index:05d}.npy"
            rejected_path = shard_dir / f"rejected_input_ids-{shard_index:05d}.npy"
            save_buffer_rows(chosen_path, chosen_rows)
            save_buffer_rows(rejected_path, rejected_rows)
            shards.append(
                {
                    "chosen_input_ids_path": str(chosen_path),
                    "rejected_input_ids_path": str(rejected_path),
                    "rows": len(chosen_rows),
                }
            )
            message_prefix = "final shard" if final else "shard"
            _progress(
                f"WebbGPT: preparing {stage}: wrote {message_prefix} {shard_index + 1} "
                f"({num_examples:,} preference examples so far)."
            )
            chosen_rows = []
            rejected_rows = []
            shard_index += 1
            snapshot_state()

        for source_index, source in enumerate(sources):
            progress = source_progress[source_index]
            accepted_records = int(progress.get("accepted_records", 0))
            _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(
                source,
                raw_records_consumed=int(progress.get("raw_records_consumed", 0)),
            ):
                progress["raw_records_consumed"] = int(progress.get("raw_records_consumed", 0)) + 1
                consumed_since_snapshot += 1
                prompt = _coerce_prompt_messages(item.get(source.prompt_field))
                chosen = item.get(source.chosen_field)
                rejected = item.get(source.rejected_field)
                if prompt is None or not isinstance(chosen, str) or not isinstance(rejected, str):
                    if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                        snapshot_state()
                    continue
                chosen_rows.append(encode_preference_example(prompt, chosen, tokenizer, self.config.sequence_length))
                rejected_rows.append(encode_preference_example(prompt, rejected, tokenizer, self.config.sequence_length))
                accepted_records += 1
                progress["accepted_records"] = accepted_records
                num_examples += 1
                if len(chosen_rows) >= self.config.prepared_shard_size:
                    flush_completed_shard()
                if accepted_records % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {accepted_records:,} preference examples so far."
                    )
                if consumed_since_snapshot >= PREPARE_EXAMPLE_SNAPSHOT_INTERVAL:
                    snapshot_state()
            _progress(
                f"WebbGPT: preparing {stage} source {source.name}: "
                f"finished with {accepted_records:,} preference examples."
            )

        flush_completed_shard(final=True)
        manifest = {
            "version": "1.0",
            "stage": stage,
            "kind": "preference",
            "input_fingerprint": input_fingerprint,
            "tokenizer_path": self.config.tokenizer_path,
            "sequence_length": self.config.sequence_length,
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "num_examples": num_examples,
            "source_snapshots": source_snapshots,
            "shards": shards,
        }
        save_prepared_manifest(manifest_path, manifest)
        remove_resume_artifacts(manifest_path)
        _progress(
            f"WebbGPT: finished preparing {stage} "
            f"({num_examples:,} examples across {len(shards):,} shards)."
        )
        return manifest

    def prepare_stage(self, stage: str, output_path: str, *, force_rebuild: bool = False) -> dict[str, Any]:
        sources = self._stage_sources(stage)
        self._require_stage_sources(stage, sources)
        _progress(
            f"WebbGPT: preparing stage {stage} from {len(sources)} source(s): "
            + ", ".join(source.name for source in sources)
        )
        if self._uses_prepared_sources(sources):
            manifest = load_prepared_manifest(sources[0].path)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            save_prepared_manifest(output, manifest)
            return manifest

        if stage in {"pretrain", "continue", "validation"}:
            return self._prepare_packed_stage(stage, sources, output_path, force_rebuild=force_rebuild)
        if stage == "sft":
            return self._prepare_sft_stage(stage, sources, output_path, force_rebuild=force_rebuild)
        if stage == "preference":
            return self._prepare_preference_stage(stage, sources, output_path, force_rebuild=force_rebuild)

        raise ValueError(f"Unsupported stage {stage!r}")

    def _load_source_records(
        self,
        source: DataSourceConfig,
        *,
        raw_records_consumed: int = 0,
    ) -> Iterable[dict[str, Any]]:
        dataset_cls, iterable_cls, load_dataset = _require_datasets()
        source = _source_with_cursor(source, raw_records_consumed)
        dataset = None
        if source.format == "hf":
            dataset_name = source.dataset_name or source.path
            if not dataset_name:
                raise ValueError(f"HF source {source.name!r} requires dataset_name or path.")
            dataset = load_dataset(
                dataset_name,
                source.dataset_config_name,
                split=source.split,
                revision=source.dataset_revision,
                streaming=bool(source.streaming),
            )
        elif source.format == "text":
            dataset = load_dataset(
                "text",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "jsonl":
            dataset = load_dataset(
                "json",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "parquet":
            dataset = load_dataset(
                "parquet",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "arrow":
            dataset = load_dataset(
                "arrow",
                data_files=_data_files_for_source(source),
                split=source.split,
                streaming=bool(source.streaming),
            )
        elif source.format == "prepared":
            raise RuntimeError("Prepared-manifest sources cannot be read as raw records.")
        else:
            raise ValueError(f"Unsupported source format {source.format}")

        dataset = _apply_record_window(dataset, source)
        if isinstance(dataset, (dataset_cls, iterable_cls)):
            return dataset
        return dataset

    def _iter_documents(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[DocumentRecord]:
        seen_hashes: set[str] = set()
        for source in sources:
            yielded_records = 0
            if stage is not None:
                location = source.dataset_name or source.path or ",".join(source.paths)
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name} "
                    f"({source.format}) from {location}."
                )
            for item in self._load_source_records(source):
                text = item.get(source.text_field, "")
                if not isinstance(text, str):
                    continue
                record = DocumentRecord(
                    text=text,
                    source=source.name,
                    metadata={field: item.get(field) for field in source.metadata_fields},
                )
                cleaned = clean_document(record, self.config, source, seen_hashes)
                if cleaned.record is not None:
                    yielded_records += 1
                    if stage is not None and yielded_records % 1000 == 0:
                        _progress(
                            f"WebbGPT: preparing {stage} source {source.name}: "
                            f"kept {yielded_records:,} documents so far."
                        )
                    yield cleaned.record
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_records:,} documents kept."
                )

    def _iter_sft_examples(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[SFTExample]:
        for source in sources:
            yielded_examples = 0
            if stage is not None:
                _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(source):
                messages = item.get(source.messages_field)
                if not isinstance(messages, list):
                    prompt_messages = _coerce_prompt_messages(item.get(source.prompt_field))
                    response = item.get(source.response_field)
                    if not isinstance(response, str):
                        response = item.get(source.chosen_field)
                    if prompt_messages is None or not isinstance(response, str):
                        continue
                    messages = [
                        *prompt_messages,
                        {"role": "assistant", "content": response},
                    ]
                yielded_examples += 1
                if stage is not None and yielded_examples % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {yielded_examples:,} SFT examples so far."
                    )
                yield SFTExample(
                    messages=messages,
                    source=source.name,
                    example_id=_message_example_id(source, item, messages),
                    split_group_id=_message_group_id(source, item, messages),
                    metadata={field: item.get(field) for field in source.metadata_fields},
                )
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_examples:,} SFT examples."
                )

    def _iter_preference_examples(
        self, sources: list[DataSourceConfig], *, stage: str | None = None
    ) -> Iterable[PreferenceExample]:
        for source in sources:
            yielded_examples = 0
            if stage is not None:
                _progress(f"WebbGPT: preparing {stage} source {source.name} ({source.format}).")
            for item in self._load_source_records(source):
                prompt = _coerce_prompt_messages(item.get(source.prompt_field))
                chosen = item.get(source.chosen_field)
                rejected = item.get(source.rejected_field)
                if prompt is None or not isinstance(chosen, str) or not isinstance(rejected, str):
                    continue
                yielded_examples += 1
                if stage is not None and yielded_examples % 500 == 0:
                    _progress(
                        f"WebbGPT: preparing {stage} source {source.name}: "
                        f"loaded {yielded_examples:,} preference examples so far."
                    )
                yield PreferenceExample(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    source=source.name,
                    example_id=_preference_example_id(source, item, prompt, chosen, rejected),
                    split_group_id=_message_group_id(source, item, prompt),
                    metadata={field: item.get(field) for field in source.metadata_fields},
                )
            if stage is not None:
                _progress(
                    f"WebbGPT: preparing {stage} source {source.name}: "
                    f"finished with {yielded_examples:,} preference examples."
                )

    def build_pretrain(self):
        self._require_stage_sources("pretrain", self.config.pretrain_sources)
        if self._uses_prepared_sources(self.config.pretrain_sources):
            return self._build_prepared_dataset(self.config.pretrain_sources, expected_kind="packed_lm")
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.pretrain_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def build_continued_pretrain(self):
        self._require_stage_sources("continue", self.config.continued_pretrain_sources)
        if self._uses_prepared_sources(self.config.continued_pretrain_sources):
            return self._build_prepared_dataset(
                self.config.continued_pretrain_sources, expected_kind="packed_lm"
            )
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.continued_pretrain_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def _build_sft_dataset_from_sources(self, sources: list[DataSourceConfig]):
        if self._uses_prepared_sources(sources):
            return self._build_prepared_dataset(sources, expected_kind="sft")
        return SFTDataset(
            list(self._iter_sft_examples(sources)),
            self.config.tokenizer_path,
            self.config.sequence_length,
        )

    def build_sft(self):
        self._require_stage_sources("sft", self.config.sft_sources)
        return self._build_sft_dataset_from_sources(self.config.sft_sources)

    def build_sft_split(
        self,
        *,
        seed: int,
        validation_fraction: float,
        validation_min_examples: int,
        allow_weak_validation: bool,
        require_explicit_validation: bool = False,
    ):
        self._require_stage_sources("sft", self.config.sft_sources)
        if self.config.sft_validation_sources:
            return (
                self._build_sft_dataset_from_sources(self.config.sft_sources),
                self._build_sft_dataset_from_sources(self.config.sft_validation_sources),
            )
        if require_explicit_validation:
            raise RuntimeError(
                "WebbGPT: local-MVP SFT requires explicit sft_validation_sources by default. "
                "Only exploratory runs may rely on grouped auto-split."
            )
        if self._uses_prepared_sources(self.config.sft_sources):
            raise RuntimeError(
                "Prepared SFT training sources require explicit sft_validation_sources in v1 because grouped auto-splitting "
                "needs access to raw prompt metadata."
            )
        examples = list(self._iter_sft_examples(self.config.sft_sources))
        train_indices, validation_indices = _split_indices_by_group(
            examples,
            stage_name="sft",
            seed=seed,
            validation_fraction=validation_fraction,
            validation_min_examples=validation_min_examples,
            allow_weak_validation=allow_weak_validation,
        )
        train_examples = [examples[index] for index in train_indices]
        validation_examples = [examples[index] for index in validation_indices]
        return (
            SFTDataset(train_examples, self.config.tokenizer_path, self.config.sequence_length),
            SFTDataset(validation_examples, self.config.tokenizer_path, self.config.sequence_length),
        )

    def _build_preference_dataset_from_sources(self, sources: list[DataSourceConfig]):
        if self._uses_prepared_sources(sources):
            return self._build_prepared_dataset(
                sources, expected_kind="preference"
            )
        return PreferenceDataset(
            list(self._iter_preference_examples(sources)),
            self.config.tokenizer_path,
            self.config.sequence_length,
        )

    def build_preference(self):
        self._require_stage_sources("preference", self.config.preference_sources)
        return self._build_preference_dataset_from_sources(self.config.preference_sources)

    def build_preference_split(
        self,
        *,
        seed: int,
        validation_fraction: float,
        validation_min_examples: int,
        allow_weak_validation: bool,
        require_explicit_validation: bool = False,
    ):
        self._require_stage_sources("preference", self.config.preference_sources)
        if self.config.preference_validation_sources:
            return (
                self._build_preference_dataset_from_sources(self.config.preference_sources),
                self._build_preference_dataset_from_sources(self.config.preference_validation_sources),
            )
        if require_explicit_validation:
            raise RuntimeError(
                "WebbGPT: local-MVP DPO requires explicit preference_validation_sources by default. "
                "Only exploratory runs may rely on grouped auto-split."
            )
        if self._uses_prepared_sources(self.config.preference_sources):
            raise RuntimeError(
                "Prepared preference training sources require explicit preference_validation_sources in v1 because grouped "
                "auto-splitting needs access to raw prompt metadata."
            )
        examples = list(self._iter_preference_examples(self.config.preference_sources))
        train_indices, validation_indices = _split_indices_by_group(
            examples,
            stage_name="preference",
            seed=seed,
            validation_fraction=validation_fraction,
            validation_min_examples=validation_min_examples,
            allow_weak_validation=allow_weak_validation,
        )
        train_examples = [examples[index] for index in train_indices]
        validation_examples = [examples[index] for index in validation_indices]
        return (
            PreferenceDataset(train_examples, self.config.tokenizer_path, self.config.sequence_length),
            PreferenceDataset(validation_examples, self.config.tokenizer_path, self.config.sequence_length),
        )

    def build_validation(self):
        self._require_stage_sources("validation", self.config.validation_sources)
        if self._uses_prepared_sources(self.config.validation_sources):
            return self._build_prepared_dataset(self.config.validation_sources, expected_kind="packed_lm")
        tokenizer = SentencePieceTokenizer(self.config.tokenizer_path)
        tokenized = (
            tokenizer.encode(doc.text, add_bos=True, add_eos=True)
            for doc in self._iter_documents(self.config.validation_sources)
        )
        sequences = pack_token_sequences(
            tokenized,
            sequence_length=self.config.sequence_length,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>"),
        )
        return PackedSequenceDataset(sequences, tokenizer.token_to_id("<pad>"))

    def export_examples(self, stage: str, output_path: str) -> None:
        sources = self._stage_sources(stage)
        self._require_stage_sources(stage, sources)
        if self._uses_prepared_sources(sources):
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(load_prepared_manifest(sources[0].path), indent=2))
            return
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if stage in {"pretrain", "continue", "validation"}:
            rows = [asdict(record) for record in self._iter_documents(sources)]
        elif stage == "sft":
            rows = [asdict(record) for record in self._iter_sft_examples(sources)]
        elif stage == "preference":
            rows = [asdict(record) for record in self._iter_preference_examples(sources)]
        else:
            raise ValueError(f"Unsupported stage {stage!r}")
        output.write_text("\n".join(json.dumps(row) for row in rows))
