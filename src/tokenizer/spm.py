from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Iterable

from config import TokenizerConfig


def _require_sentencepiece():
    try:
        import sentencepiece as spm  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentencepiece is required for tokenizer training and inference. "
            "Install it with `pip install sentencepiece`."
        ) from exc
    return spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        spm = _require_sentencepiece()
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return list(self.processor.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos))

    def decode(self, ids: Iterable[int]) -> str:
        return str(self.processor.decode(list(ids)))

    def token_to_id(self, token: str) -> int:
        return int(self.processor.piece_to_id(token))

    def id_to_token(self, idx: int) -> str:
        return str(self.processor.id_to_piece(idx))


def _describe_input_files(input_files: list[str]) -> tuple[int, int]:
    existing_files = 0
    total_bytes = 0
    for raw_path in input_files:
        path = Path(raw_path)
        if not path.exists():
            continue
        existing_files += 1
        total_bytes += path.stat().st_size
    return existing_files, total_bytes


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _heartbeat(stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.wait(interval_seconds):
        print(
            "WebbGPT: tokenizer training is still running. SentencePiece can stay quiet for long stretches during BPE optimization.",
            file=sys.stderr,
            flush=True,
        )


def train_tokenizer(
    input_files: list[str],
    config: TokenizerConfig,
    user_defined_symbols: list[str] | None = None,
) -> Path:
    spm = _require_sentencepiece()
    model_prefix = Path(config.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    reserved_symbols = {
        config.special_tokens["unk_token"],
        config.special_tokens["bos_token"],
        config.special_tokens["eos_token"],
        config.special_tokens["pad_token"],
    }
    symbols = user_defined_symbols or [
        token for token in config.special_tokens.values() if token not in reserved_symbols
    ]
    args = {
        "input": ",".join(input_files),
        "model_prefix": str(model_prefix),
        "vocab_size": config.vocab_size,
        "model_type": config.model_type,
        "character_coverage": config.character_coverage,
        "byte_fallback": str(config.byte_fallback).lower(),
        "normalization_rule_name": config.normalization_rule_name,
        # SentencePiece expects `input_sentence_size`; keep the config field name
        # as-is and translate it here so older saved configs still work.
        "input_sentence_size": config.sample_input_sentence_size,
        "max_sentence_length": config.max_sentence_length,
        "train_extremely_large_corpus": str(config.train_extremely_large_corpus).lower(),
        "user_defined_symbols": ",".join(symbols),
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "unk_piece": config.special_tokens["unk_token"],
        "pad_piece": config.special_tokens["pad_token"],
        "bos_piece": config.special_tokens["bos_token"],
        "eos_piece": config.special_tokens["eos_token"],
    }
    command = " ".join(f"--{key}={value}" for key, value in args.items())
    existing_files, total_bytes = _describe_input_files(input_files)
    print(
        "WebbGPT: starting tokenizer training "
        f"on {existing_files} file(s), {_format_bytes(total_bytes)} total. "
        "SentencePiece may go quiet for a while after its initial logs; that is normal.",
        file=sys.stderr,
        flush=True,
    )
    stop_event = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event, 20.0), daemon=True)
    heartbeat.start()
    try:
        spm.SentencePieceTrainer.Train(command)
    except RuntimeError as exc:
        message = str(exc)
        if "Vocabulary size too high" in message:
            total_chars = 0
            total_lines = 0
            total_words = 0
            for raw_path in input_files:
                path = Path(raw_path)
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                total_chars += len(text)
                total_lines += len(text.splitlines())
                total_words += len(text.split())
            raise RuntimeError(
                "Tokenizer training failed because the requested vocab size is much larger than the "
                "available corpus can support. "
                f"Requested vocab size: {config.vocab_size}. "
                f"Observed corpus size: {total_lines} lines, {total_words} words, {total_chars} characters. "
                "If you want the real tokenizer at 50176, provide a much larger corpus file or corpus shard list."
            ) from exc
        raise
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    meta_path = model_prefix.with_suffix(".tokenizer.json")
    meta_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(
        f"WebbGPT: tokenizer training finished. Wrote {model_prefix.with_suffix('.model')} and {meta_path}.",
        file=sys.stderr,
        flush=True,
    )
    return model_prefix.with_suffix(".model")
