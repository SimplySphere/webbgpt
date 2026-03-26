from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from sqlalchemy import Select, create_engine, or_, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from grounding.sql.models import (
    AdmissionsFact,
    AthleticsGame,
    AthleticsRecord,
    AthleticsTeam,
    Base,
    Course,
    CourseVersion,
    FacultyProfile,
    HandbookSection,
    Institution,
    KnowledgeSnapshot,
    Program,
    PublicationVersion,
    RetrievalChunk,
    Section,
    SourceDocument,
    Term,
)


WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split()).strip().lower()


def _tokenize(value: str | None) -> list[str]:
    return WORD_RE.findall(_normalize_text(value))


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_field_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_field_text(item)}" for key, item in value.items())
    return str(value)


def _text_score(query: str, *fields: Any) -> float:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0.0
    query_tokens = set(_tokenize(normalized_query))
    haystacks = [_normalize_text(_field_text(field)) for field in fields if _field_text(field)]
    if not haystacks:
        return 0.0
    score = 0.0
    for haystack in haystacks:
        if normalized_query in haystack:
            score += 8.0
        hay_tokens = set(_tokenize(haystack))
        score += 1.5 * len(query_tokens & hay_tokens)
        if len(normalized_query) >= 5:
            score += SequenceMatcher(None, normalized_query, haystack[: max(len(normalized_query) * 2, 32)]).ratio()
    return score


class _BaseStore:
    def __init__(self, dsn: str):
        self._prepare_sqlite_path(dsn)
        self.engine = create_engine(dsn, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _prepare_sqlite_path(dsn: str) -> None:
        url = make_url(dsn)
        if url.drivername != "sqlite":
            return
        database = url.database
        if not database or database == ":memory:":
            return
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self.session_factory()

    def _list(self, stmt: Select) -> list:
        with self.session() as session:
            return list(session.scalars(stmt))


class CatalogStore(_BaseStore):
    def search_courses(self, query: str, institution_id: str | None = None, limit: int = 10) -> list[Course]:
        stmt = select(Course).where(
            or_(
                Course.code.ilike(f"%{query}%"),
                Course.title.ilike(f"%{query}%"),
                Course.description.ilike(f"%{query}%"),
            )
        )
        if institution_id is not None:
            stmt = stmt.where(Course.institution_id == institution_id)
        return self._list(stmt.limit(limit))

    def get_course(self, course_id: str) -> Course | None:
        with self.session() as session:
            return session.get(Course, course_id)

    def get_program(self, program_id: str) -> Program | None:
        with self.session() as session:
            return session.get(Program, program_id)

    def resolve_prereqs(self, course_id: str) -> dict | None:
        course = self.get_course(course_id)
        return None if course is None else course.prerequisites

    def current_sections(self, course_id: str, term_id: str | None = None) -> list[Section]:
        stmt = select(Section).where(Section.course_id == course_id)
        if term_id is not None:
            stmt = stmt.where(Section.term_id == term_id)
        return self._list(stmt)

    def upsert_institutions(self, rows: Iterable[dict]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(Institution(**row))
            session.commit()

    def upsert_terms(self, rows: Iterable[dict]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(Term(**row))
            session.commit()

    def upsert_programs(self, rows: Iterable[dict]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(Program(**row))
            session.commit()

    def upsert_courses(self, rows: Iterable[dict]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(Course(**row))
            session.commit()

    def upsert_sections(self, rows: Iterable[dict]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(Section(**row))
            session.commit()


class WebbKnowledgeStore(_BaseStore):
    FAMILY_TABLE_MAP = {
        "course_catalog": [CourseVersion],
        "faculty": [FacultyProfile],
        "handbook_policy": [HandbookSection],
        "admissions_general": [AdmissionsFact],
        "academic_publications": [PublicationVersion],
        "athletics": [AthleticsTeam, AthleticsGame, AthleticsRecord],
    }

    def create_snapshot(
        self,
        *,
        label: str,
        seed_url_pack: str | None = None,
        handbook_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        snapshot_id: str | None = None,
    ) -> str:
        snapshot_key = snapshot_id or _stable_id(
            "snapshot",
            {"label": label, "seed_url_pack": seed_url_pack, "handbook_url": handbook_url, "metadata": metadata or {}},
        )
        row = {
            "id": snapshot_key,
            "label": label,
            "source_family": "webb",
            "created_at": metadata.get("created_at", "") if metadata else "",
            "completed": False,
            "current": False,
            "seed_url_pack": seed_url_pack,
            "handbook_url": handbook_url,
            "extra": metadata or {},
        }
        with self.session() as session:
            session.merge(KnowledgeSnapshot(**row))
            session.commit()
        return snapshot_key

    def complete_snapshot(self, snapshot_id: str, *, metadata: dict[str, Any] | None = None) -> KnowledgeSnapshot:
        with self.session() as session:
            snapshots = list(session.scalars(select(KnowledgeSnapshot)))
            for snapshot in snapshots:
                snapshot.current = snapshot.id == snapshot_id
                if snapshot.id == snapshot_id:
                    snapshot.completed = True
                    merged_metadata = dict(snapshot.extra or {})
                    if metadata:
                        merged_metadata.update(metadata)
                    snapshot.extra = merged_metadata
            self._set_current_flags(session, SourceDocument, snapshot_id)
            self._set_current_flags(session, RetrievalChunk, snapshot_id)
            self._set_current_flags(session, CourseVersion, snapshot_id)
            self._set_current_flags(session, FacultyProfile, snapshot_id)
            self._set_current_flags(session, HandbookSection, snapshot_id)
            self._set_current_flags(session, AdmissionsFact, snapshot_id)
            self._set_current_flags(session, PublicationVersion, snapshot_id)
            self._set_current_flags(session, AthleticsTeam, snapshot_id)
            self._set_current_flags(session, AthleticsGame, snapshot_id)
            self._set_current_flags(session, AthleticsRecord, snapshot_id)
            session.commit()
            snapshot = session.get(KnowledgeSnapshot, snapshot_id)
            if snapshot is None:
                raise RuntimeError(f"Snapshot {snapshot_id!r} was not found after completion.")
            return snapshot

    @staticmethod
    def _set_current_flags(session: Session, model, snapshot_id: str) -> None:
        rows = list(session.scalars(select(model)))
        for row in rows:
            row.current = row.snapshot_id == snapshot_id

    def latest_snapshot(self) -> KnowledgeSnapshot | None:
        with self.session() as session:
            current = session.scalars(
                select(KnowledgeSnapshot).where(KnowledgeSnapshot.current.is_(True))
            ).first()
            if current is not None:
                return current
            snapshots = list(session.scalars(select(KnowledgeSnapshot)))
            if not snapshots:
                return None
            return sorted(snapshots, key=lambda row: (row.created_at or "", row.id))[-1]

    def latest_completed_snapshot(self) -> KnowledgeSnapshot | None:
        with self.session() as session:
            current = session.scalars(
                select(KnowledgeSnapshot).where(
                    KnowledgeSnapshot.current.is_(True),
                    KnowledgeSnapshot.completed.is_(True),
                )
            ).first()
            if current is not None:
                return current
            snapshots = list(
                session.scalars(select(KnowledgeSnapshot).where(KnowledgeSnapshot.completed.is_(True)))
            )
            if not snapshots:
                return None
            return sorted(snapshots, key=lambda row: (row.created_at or "", row.id))[-1]

    def resolve_snapshot_id(self, snapshot_id: str | None) -> str | None:
        if snapshot_id and snapshot_id != "latest":
            return snapshot_id
        snapshot = self.latest_completed_snapshot() or self.latest_snapshot()
        return None if snapshot is None else snapshot.id

    def get_snapshot(self, snapshot_id: str | None) -> KnowledgeSnapshot | None:
        resolved = self.resolve_snapshot_id(snapshot_id)
        if resolved is None:
            return None
        with self.session() as session:
            return session.get(KnowledgeSnapshot, resolved)

    def _upsert_model_rows(self, model, rows: Iterable[dict[str, Any]]) -> None:
        with self.session() as session:
            for row in rows:
                session.merge(model(**row))
            session.commit()

    def upsert_source_documents(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(SourceDocument, rows)

    def upsert_retrieval_chunks(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(RetrievalChunk, rows)

    def upsert_course_versions(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(CourseVersion, rows)

    def upsert_faculty_profiles(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(FacultyProfile, rows)

    def upsert_handbook_sections(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(HandbookSection, rows)

    def upsert_admissions_facts(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(AdmissionsFact, rows)

    def upsert_publication_versions(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(PublicationVersion, rows)

    def upsert_athletics_teams(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(AthleticsTeam, rows)

    def upsert_athletics_games(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(AthleticsGame, rows)

    def upsert_athletics_records(self, rows: Iterable[dict[str, Any]]) -> None:
        self._upsert_model_rows(AthleticsRecord, rows)

    def _snapshot_rows(self, model, snapshot_id: str | None) -> list:
        resolved = self.resolve_snapshot_id(snapshot_id)
        if resolved is None:
            return []
        stmt = select(model).where(model.snapshot_id == resolved)
        return self._list(stmt)

    def list_source_documents(self, snapshot_id: str | None = None) -> list[SourceDocument]:
        return self._snapshot_rows(SourceDocument, snapshot_id)

    def snapshot_family_metadata(self, snapshot_id: str | None = None) -> dict[str, Any]:
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            return {}
        return dict((snapshot.extra or {}).get("families") or {})

    def snapshot_quality(self, snapshot_id: str | None = None) -> dict[str, Any]:
        sections = self._snapshot_rows(HandbookSection, snapshot_id)
        snapshot = self.get_snapshot(snapshot_id)
        family_metadata = self.snapshot_family_metadata(snapshot_id)
        if not sections:
            return {
                "handbook_citable": False,
                "handbook_extraction_modes": [],
                "family_metadata": family_metadata,
                "snapshot_completed": bool(snapshot.completed) if snapshot is not None else False,
            }
        extraction_modes = sorted({section.extraction_quality for section in sections})
        return {
            "handbook_citable": all(mode not in {"degraded", "uncitable"} for mode in extraction_modes),
            "handbook_extraction_modes": extraction_modes,
            "family_metadata": family_metadata,
            "snapshot_completed": bool(snapshot.completed) if snapshot is not None else False,
        }

    @staticmethod
    def _row_payload(row: Any) -> dict[str, Any]:
        return {attribute.key: getattr(row, attribute.key) for attribute in row.__mapper__.column_attrs}

    def carry_forward_families(
        self,
        from_snapshot_id: str | None,
        to_snapshot_id: str,
        *,
        families: list[str],
    ) -> dict[str, int]:
        resolved_from = self.resolve_snapshot_id(from_snapshot_id)
        if resolved_from is None or not families:
            return {}
        family_set = set(families)
        carried_counts: dict[str, int] = {family: 0 for family in family_set}
        with self.session() as session:
            old_documents = list(
                session.scalars(
                    select(SourceDocument).where(
                        SourceDocument.snapshot_id == resolved_from,
                        SourceDocument.source_family.in_(family_set),
                    )
                )
            )
            doc_id_map: dict[str, str] = {}
            for row in old_documents:
                payload = self._row_payload(row)
                payload["id"] = _stable_id("document", [to_snapshot_id, row.id])
                doc_id_map[row.id] = payload["id"]
                payload["snapshot_id"] = to_snapshot_id
                payload["current"] = False
                session.merge(SourceDocument(**_json_clone(payload)))
                carried_counts[row.source_family] = carried_counts.get(row.source_family, 0) + 1
            old_chunks = list(
                session.scalars(
                    select(RetrievalChunk).where(
                        RetrievalChunk.snapshot_id == resolved_from,
                        RetrievalChunk.source_family.in_(family_set),
                    )
                )
            )
            for row in old_chunks:
                payload = self._row_payload(row)
                payload["id"] = _stable_id("chunk", [to_snapshot_id, row.id])
                payload["snapshot_id"] = to_snapshot_id
                payload["document_id"] = doc_id_map.get(row.document_id, row.document_id)
                payload["current"] = False
                session.merge(RetrievalChunk(**_json_clone(payload)))
            for family in family_set:
                for model in self.FAMILY_TABLE_MAP.get(family, []):
                    rows = list(
                        session.scalars(select(model).where(model.snapshot_id == resolved_from))
                    )
                    for row in rows:
                        payload = self._row_payload(row)
                        payload["id"] = _stable_id(model.__tablename__, [to_snapshot_id, row.id])
                        payload["snapshot_id"] = to_snapshot_id
                        payload["current"] = False
                        session.merge(model(**_json_clone(payload)))
                        carried_counts[family] = carried_counts.get(family, 0) + 1
            session.commit()
        return carried_counts

    def _rank_rows(self, query: str, rows: list[Any], *field_names: str, limit: int = 5) -> list[Any]:
        scored: list[tuple[float, Any]] = []
        for row in rows:
            score = _text_score(query, *[getattr(row, field_name, None) for field_name in field_names])
            if score <= 0:
                continue
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def search_course_versions(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        school_year: str | None = None,
        limit: int = 5,
    ) -> list[CourseVersion]:
        rows = self._snapshot_rows(CourseVersion, snapshot_id)
        if school_year is not None:
            rows = [row for row in rows if row.school_year == school_year]
        return self._rank_rows(
            query,
            rows,
            "course_code",
            "title",
            "department",
            "description",
            "prerequisites",
            "teacher_names",
            limit=limit,
        )

    def search_faculty_profiles(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        limit: int = 5,
    ) -> list[FacultyProfile]:
        rows = self._snapshot_rows(FacultyProfile, snapshot_id)
        return self._rank_rows(query, rows, "name", "role_title", "department", "bio", limit=limit)

    def search_handbook_sections(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        limit: int = 5,
    ) -> list[HandbookSection]:
        rows = self._snapshot_rows(HandbookSection, snapshot_id)
        return self._rank_rows(query, rows, "section_title", "section_path", "text", limit=limit)

    def search_admissions_facts(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        limit: int = 5,
    ) -> list[AdmissionsFact]:
        rows = self._snapshot_rows(AdmissionsFact, snapshot_id)
        return self._rank_rows(query, rows, "title", "text", limit=limit)

    def search_publications(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        limit: int = 5,
    ) -> list[PublicationVersion]:
        rows = self._snapshot_rows(PublicationVersion, snapshot_id)
        return self._rank_rows(query, rows, "title", "school_year", "text", limit=limit)

    def search_retrieval_chunks(
        self,
        query: str,
        *,
        source_types: list[str] | None = None,
        source_families: list[str] | None = None,
        snapshot_id: str | None = None,
        limit: int = 5,
    ) -> list[RetrievalChunk]:
        rows = self._snapshot_rows(RetrievalChunk, snapshot_id)
        if source_types:
            allowed = set(source_types)
            rows = [row for row in rows if row.source_type in allowed]
        if source_families:
            allowed_families = set(source_families)
            rows = [row for row in rows if row.source_family in allowed_families]
        return self._rank_rows(query, rows, "heading", "text", "normalized_text", limit=limit)

    def search_athletics_teams(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        season: str | None = None,
        limit: int = 5,
    ) -> list[AthleticsTeam]:
        rows = self._snapshot_rows(AthleticsTeam, snapshot_id)
        if season is not None:
            rows = [row for row in rows if row.season == season]
        return self._rank_rows(query, rows, "team_name", "coach_name", "summary", limit=limit)

    def search_athletics_games(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        season: str | None = None,
        limit: int = 5,
    ) -> list[AthleticsGame]:
        rows = self._snapshot_rows(AthleticsGame, snapshot_id)
        if season is not None:
            rows = [row for row in rows if row.season == season]
        return self._rank_rows(query, rows, "team_name", "opponent", "game_date", "result", "venue", limit=limit)

    def search_athletics_records(
        self,
        query: str,
        *,
        snapshot_id: str | None = None,
        season: str | None = None,
        limit: int = 5,
    ) -> list[AthleticsRecord]:
        rows = self._snapshot_rows(AthleticsRecord, snapshot_id)
        if season is not None:
            rows = [row for row in rows if row.season == season]
        return self._rank_rows(query, rows, "team_name", "record_type", "value", limit=limit)

    def compare_course_versions(
        self,
        query: str,
        *,
        school_year_a: str,
        school_year_b: str,
        snapshot_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, list[CourseVersion]]:
        left_rows = self.search_course_versions(query, snapshot_id=snapshot_id, school_year=school_year_a, limit=200)
        right_rows = self.search_course_versions(query, snapshot_id=snapshot_id, school_year=school_year_b, limit=200)
        left_by_key = {row.course_key: row for row in left_rows}
        right_by_key = {row.course_key: row for row in right_rows}
        added = [right_by_key[key] for key in right_by_key.keys() - left_by_key.keys()]
        removed = [left_by_key[key] for key in left_by_key.keys() - right_by_key.keys()]
        changed: list[CourseVersion] = []
        for key in left_by_key.keys() & right_by_key.keys():
            left = left_by_key[key]
            right = right_by_key[key]
            if any(
                [
                    _normalize_text(left.title) != _normalize_text(right.title),
                    _normalize_text(left.description) != _normalize_text(right.description),
                    _normalize_text(left.prerequisites) != _normalize_text(right.prerequisites),
                    _normalize_text(_field_text(left.teacher_names)) != _normalize_text(_field_text(right.teacher_names)),
                ]
            ):
                changed.append(right)
        return {
            "added": added[:limit],
            "removed": removed[:limit],
            "changed": changed[:limit],
        }

    def diff_snapshots(self, snapshot_from: str, snapshot_to: str) -> dict[str, Any]:
        before = {row.source_url: row for row in self.list_source_documents(snapshot_from)}
        after = {row.source_url: row for row in self.list_source_documents(snapshot_to)}
        added = sorted(after.keys() - before.keys())
        removed = sorted(before.keys() - after.keys())
        changed = sorted(
            url
            for url in before.keys() & after.keys()
            if before[url].content_hash != after[url].content_hash
        )
        unchanged = sorted(
            url
            for url in before.keys() & after.keys()
            if before[url].content_hash == after[url].content_hash
        )
        return {
            "from_snapshot": snapshot_from,
            "to_snapshot": snapshot_to,
            "added_urls": added,
            "removed_urls": removed,
            "changed_urls": changed,
            "unchanged_urls": unchanged,
            "counts": {
                "added": len(added),
                "removed": len(removed),
                "changed": len(changed),
                "unchanged": len(unchanged),
            },
        }
