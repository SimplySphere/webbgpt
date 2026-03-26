from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from config import TokenizerCorpusConfig


WHITESPACE_RE = re.compile(r"\s+")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is required to build the tokenizer corpus. Install it with `pip install datasets`."
        ) from exc
    return load_dataset


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def build_tokenizer_corpus(config: TokenizerCorpusConfig) -> dict[str, int | str]:
    load_dataset = _require_datasets()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(
        f"WebbGPT: building tokenizer corpus from {config.dataset_name} "
        f"({config.dataset_config_name}, split={config.split})."
    )

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
        streaming=config.streaming,
    )

    documents_written = 0
    characters_written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(config.text_field)
            if not isinstance(text, str):
                continue
            if config.normalize_whitespace:
                text = _normalize_text(text)
            if len(text) < config.min_document_chars:
                continue

            handle.write(text)
            handle.write("\n")
            documents_written += 1
            characters_written += len(text) + 1

            if documents_written % 10000 == 0:
                _progress(
                    f"WebbGPT: tokenizer corpus build is still running "
                    f"({documents_written:,} documents, {characters_written:,} characters written)."
                )

            if documents_written >= config.max_documents:
                break
            if characters_written >= config.max_characters:
                break

    metadata = {
        "config": config.to_dict(),
        "dataset_name": config.dataset_name,
        "dataset_config_name": config.dataset_config_name,
        "split": config.split,
        "text_field": config.text_field,
        "output_path": str(output_path),
        "streaming": config.streaming,
        "documents_written": documents_written,
        "characters_written": characters_written,
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2))
    _progress(
        f"WebbGPT: tokenizer corpus build finished "
        f"({documents_written:,} documents, {characters_written:,} characters)."
    )
    return metadata
