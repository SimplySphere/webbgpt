from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Institution(Base):
    __tablename__ = "institutions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    website: Mapped[str | None] = mapped_column(String(512), nullable=True)

    programs: Mapped[list["Program"]] = relationship(back_populates="institution")
    courses: Mapped[list["Course"]] = relationship(back_populates="institution")
    terms: Mapped[list["Term"]] = relationship(back_populates="institution")


class Term(Base):
    __tablename__ = "terms"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    institution_id: Mapped[str] = mapped_column(ForeignKey("institutions.id"), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    starts_on: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ends_on: Mapped[str | None] = mapped_column(String(32), nullable=True)

    institution: Mapped[Institution] = relationship(back_populates="terms")

    __table_args__ = (UniqueConstraint("institution_id", "code", name="uq_term_code"),)


class Program(Base):
    __tablename__ = "programs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    institution_id: Mapped[str] = mapped_column(ForeignKey("institutions.id"), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    institution: Mapped[Institution] = relationship(back_populates="programs")
    courses: Mapped[list["Course"]] = relationship(back_populates="program")

    __table_args__ = (UniqueConstraint("institution_id", "code", name="uq_program_code"),)


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    institution_id: Mapped[str] = mapped_column(ForeignKey("institutions.id"), nullable=False)
    program_id: Mapped[str | None] = mapped_column(ForeignKey("programs.id"), nullable=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    credits: Mapped[float | None] = mapped_column(nullable=True)
    prerequisites: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    institution: Mapped[Institution] = relationship(back_populates="courses")
    program: Mapped[Program | None] = relationship(back_populates="courses")
    sections: Mapped[list["Section"]] = relationship(back_populates="course")

    __table_args__ = (UniqueConstraint("institution_id", "code", name="uq_course_code"),)


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), nullable=False)
    term_id: Mapped[str] = mapped_column(ForeignKey("terms.id"), nullable=False)
    instructor: Mapped[str | None] = mapped_column(String(255), nullable=True)
    meeting_times: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    seats_total: Mapped[int | None] = mapped_column(nullable=True)
    seats_available: Mapped[int | None] = mapped_column(nullable=True)
    modality: Mapped[str | None] = mapped_column(String(64), nullable=True)

    course: Mapped[Course] = relationship(back_populates="sections")


class KnowledgeSnapshot(Base):
    __tablename__ = "knowledge_snapshots"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    source_family: Mapped[str] = mapped_column(String(64), nullable=False, default="webb")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    seed_url_pack: Mapped[str | None] = mapped_column(String(512), nullable=True)
    handbook_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    document_key: Mapped[str] = mapped_column(String(128), nullable=False)
    source_family: Mapped[str] = mapped_column(String(64), nullable=False, default="webb")
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    page_type: Mapped[str] = mapped_column(String(64), nullable=False)
    page_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    school_year: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (UniqueConstraint("snapshot_id", "source_url", name="uq_source_document_snapshot_url"),)


class RetrievalChunk(Base):
    __tablename__ = "retrieval_chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("source_documents.id"), nullable=False)
    source_family: Mapped[str] = mapped_column(String(64), nullable=False, default="webb")
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    school_year: Mapped[str | None] = mapped_column(String(32), nullable=True)
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "document_id", "chunk_index", name="uq_retrieval_chunk_snapshot_doc_idx"),
    )


class CourseVersion(Base):
    __tablename__ = "course_versions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    course_key: Mapped[str] = mapped_column(String(128), nullable=False)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    course_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    school_year: Mapped[str | None] = mapped_column(String(32), nullable=True)
    level_marker: Mapped[str | None] = mapped_column(String(128), nullable=True)
    grade_eligibility: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prerequisites: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    teacher_names: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "course_key", "school_year", name="uq_course_version_snapshot_key_year"),
    )


class FacultyProfile(Base):
    __tablename__ = "faculty_profiles"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    faculty_key: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (UniqueConstraint("snapshot_id", "faculty_key", name="uq_faculty_profile_snapshot_key"),)


class HandbookSection(Base):
    __tablename__ = "handbook_sections"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    section_key: Mapped[str] = mapped_column(String(128), nullable=False)
    handbook_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    section_title: Mapped[str] = mapped_column(String(255), nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_start: Mapped[int | None] = mapped_column(nullable=True)
    page_end: Mapped[int | None] = mapped_column(nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extraction_quality: Mapped[str] = mapped_column(String(64), nullable=False, default="native_text")
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "section_key", "page_start", name="uq_handbook_section_snapshot_key_page"),
    )


class AdmissionsFact(Base):
    __tablename__ = "admissions_facts"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    fact_key: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (UniqueConstraint("snapshot_id", "fact_key", name="uq_admissions_fact_snapshot_key"),)


class PublicationVersion(Base):
    __tablename__ = "publication_versions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    publication_key: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    school_year: Mapped[str | None] = mapped_column(String(32), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "publication_key", "school_year", name="uq_publication_snapshot_key_year"),
    )


class AthleticsTeam(Base):
    __tablename__ = "athletics_teams"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    team_key: Mapped[str] = mapped_column(String(128), nullable=False)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    coach_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "team_key", "season", name="uq_athletics_team_snapshot_key_season"),
    )


class AthleticsGame(Base):
    __tablename__ = "athletics_games"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    game_key: Mapped[str] = mapped_column(String(128), nullable=False)
    team_key: Mapped[str] = mapped_column(String(128), nullable=False)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    opponent: Mapped[str] = mapped_column(String(255), nullable=False)
    game_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result: Mapped[str | None] = mapped_column(String(255), nullable=True)
    venue: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "game_key", name="uq_athletics_game_snapshot_key"),
    )


class AthleticsRecord(Base):
    __tablename__ = "athletics_records"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("knowledge_snapshots.id"), nullable=False)
    record_key: Mapped[str] = mapped_column(String(128), nullable=False)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    record_type: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    citation_label: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    extra: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("snapshot_id", "record_key", "season", name="uq_athletics_record_snapshot_key_season"),
    )
