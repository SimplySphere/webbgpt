from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from config import DataConfig, DataSourceConfig
from data.schemas import DocumentRecord


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*)\d{3}[-.\s]*\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MULTISPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class CleanResult:
    record: DocumentRecord | None
    dropped_reason: str | None = None


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return SSN_RE.sub("[SSN]", text)


def quality_filter(text: str, config: DataConfig) -> bool:
    if len(text) < config.min_document_chars or len(text) > config.max_document_chars:
        return False
    alpha_chars = sum(char.isalpha() for char in text)
    if not text:
        return False
    if alpha_chars / max(len(text), 1) < 0.2:
        return False
    if text.count("http") > 100:
        return False
    return True


def stable_document_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def clean_document(
    record: DocumentRecord,
    data_config: DataConfig,
    source_config: DataSourceConfig,
    seen_hashes: set[str] | None = None,
) -> CleanResult:
    text = normalize_whitespace(record.text)
    if source_config.pii_scrub:
        text = scrub_pii(text)
    if source_config.quality_filter and not quality_filter(text, data_config):
        return CleanResult(record=None, dropped_reason="quality_filter")
    doc_hash = stable_document_hash(text)
    if source_config.deduplicate and seen_hashes is not None:
        if doc_hash in seen_hashes:
            return CleanResult(record=None, dropped_reason="duplicate")
        seen_hashes.add(doc_hash)
    record.text = text
    record.document_id = record.document_id or doc_hash
    return CleanResult(record=record)

