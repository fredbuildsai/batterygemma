"""Database schema.

Uses only portable SQLAlchemy types (String, Text, Integer, Float, Boolean, JSON, DateTime) so the
same models run on SQLite now and PostgreSQL/Supabase later.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --- Ingestion -------------------------------------------------------------------------------


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)  # "<source>:<external_id>"
    source: Mapped[str] = mapped_column(String(32), index=True)
    external_id: Mapped[str] = mapped_column(String(128))
    doi: Mapped[str | None] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(Text)
    norm_title: Mapped[str] = mapped_column(Text, index=True)
    authors: Mapped[list[Any]] = mapped_column(default=list)
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    xml_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(64), index=True)
    license_evidence: Mapped[str | None] = mapped_column(Text)
    license_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    relevance: Mapped[float | None] = mapped_column(Float)
    topic_tags: Mapped[list[Any]] = mapped_column(default=list)
    status: Mapped[str] = mapped_column(String(16), default="discovered", index=True)
    status_reason: Mapped[str | None] = mapped_column(Text)
    split: Mapped[str | None] = mapped_column(String(8), index=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(128))
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    files: Mapped[list["File"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True)
    kind: Mapped[str] = mapped_column(String(8))  # pdf | xml | tex
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="files")


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(200), primary_key=True)  # "<doc_id>#s<section>-cNN"
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True)
    section_path: Mapped[list[Any]] = mapped_column(default=list)
    section_type: Mapped[str | None] = mapped_column(String(32))  # intro/methods/results/discussion/...
    order: Mapped[int] = mapped_column(Integer)
    tokens: Mapped[int] = mapped_column(Integer)
    overlap_prev_tokens: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    captions: Mapped[list[Any]] = mapped_column(default=list)
    quality: Mapped[dict[str, Any]] = mapped_column(default=dict)
    purpose: Mapped[str] = mapped_column(String(8), default="sft")  # sft | cpt

    document: Mapped[Document] = relationship(back_populates="chunks")


class Material(Base):
    __tablename__ = "materials"

    material_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. Materials Project id
    source: Mapped[str] = mapped_column(String(32))
    formula: Mapped[str] = mapped_column(String(128), index=True)
    properties: Mapped[dict[str, Any]] = mapped_column(default=dict)
    license: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- Annotation and generation ------------------------------------------------------------------


class ProvenanceMixin:
    """Columns carried by every annotated or generated record."""

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_ids: Mapped[list[Any]] = mapped_column(default=list)
    chunk_ids: Mapped[list[Any]] = mapped_column(default=list)
    license: Mapped[str | None] = mapped_column(String(64))
    split: Mapped[str | None] = mapped_column(String(8), index=True)
    generator_model: Mapped[str | None] = mapped_column(String(128), index=True)
    judge_scores: Mapped[dict[str, Any]] = mapped_column(default=dict)
    tier: Mapped[str] = mapped_column(String(16), default="silver")  # silver | gold | database
    status: Mapped[str] = mapped_column(String(16), default="generated", index=True)  # generated/accepted/rejected
    reject_reason: Mapped[str | None] = mapped_column(Text)
    human_checked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Fact(ProvenanceMixin, Base):
    __tablename__ = "facts"

    material: Mapped[dict[str, Any]] = mapped_column(default=dict)  # name, formula, class, component
    component: Mapped[str | None] = mapped_column(String(32), index=True)
    property: Mapped[str] = mapped_column(Text)
    value: Mapped[str | None] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(String(64))
    conditions: Mapped[dict[str, Any]] = mapped_column(default=dict)
    category: Mapped[str | None] = mapped_column(String(32))
    polarity: Mapped[str] = mapped_column(String(16), default="positive")
    triple: Mapped[list[Any]] = mapped_column(default=list)
    evidence_sentence: Mapped[str] = mapped_column(Text)
    label_quality: Mapped[str | None] = mapped_column(String(32))


class Comparison(ProvenanceMixin, Base):
    __tablename__ = "comparisons"

    baseline: Mapped[str] = mapped_column(Text)
    modification: Mapped[str] = mapped_column(Text)
    metric: Mapped[str] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(String(16))
    magnitude: Mapped[str | None] = mapped_column(Text)
    conditions: Mapped[dict[str, Any]] = mapped_column(default=dict)
    component: Mapped[str | None] = mapped_column(String(32), index=True)
    evidence_sentence: Mapped[str] = mapped_column(Text)


class ClaimPair(ProvenanceMixin, Base):
    __tablename__ = "claim_pairs"

    sentence_1: Mapped[str] = mapped_column(Text)
    sentence_2: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(16), index=True)  # paraphrase | contradiction
    subset_name: Mapped[str] = mapped_column(String(16))  # swap | numeric | mechanism | entity
    label_quality: Mapped[str | None] = mapped_column(String(32))


class StratifiedMixin:
    """Labels used by the balancing planner and export filters."""

    task_format: Mapped[str] = mapped_column(String(32), index=True)
    polarity: Mapped[str] = mapped_column(String(8), index=True)  # positive | negative
    question_type: Mapped[str | None] = mapped_column(String(40), index=True)
    component: Mapped[str | None] = mapped_column(String(32), index=True)
    chemistry: Mapped[str | None] = mapped_column(String(64))
    grounding: Mapped[str | None] = mapped_column(String(24))  # source | source+background


class QA(ProvenanceMixin, StratifiedMixin, Base):
    __tablename__ = "qa"

    answer_type: Mapped[str] = mapped_column(String(8))  # OPEN | CLOSED
    phrase_type: Mapped[str] = mapped_column(String(16), default="free-form")
    qid_linked_id: Mapped[str | None] = mapped_column(String(64))
    closed_label: Mapped[str | None] = mapped_column(String(16))  # yes | no | option letter
    triple: Mapped[list[Any]] = mapped_column(default=list)
    system: Mapped[str | None] = mapped_column(Text)
    turns: Mapped[list[Any]] = mapped_column(default=list)  # [{"role": "user"|"assistant", "content": ...}]
    reasoning: Mapped[str | None] = mapped_column(Text)


class Ideation(ProvenanceMixin, StratifiedMixin, Base):
    __tablename__ = "ideation"

    problem: Mapped[str] = mapped_column(Text)
    constraints: Mapped[list[Any]] = mapped_column(default=list)
    reasoning: Mapped[str | None] = mapped_column(Text)
    ideas: Mapped[list[Any]] = mapped_column(default=list)
    rubric: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Negative(ProvenanceMixin, StratifiedMixin, Base):
    __tablename__ = "negatives"

    kind: Mapped[str] = mapped_column(String(32), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    flawed_element: Mapped[str | None] = mapped_column(Text)
    expert_response: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text)
    linked_positive_id: Mapped[str | None] = mapped_column(String(64))


class DPOPair(ProvenanceMixin, Base):
    __tablename__ = "dpo_pairs"

    source_qa_id: Mapped[str | None] = mapped_column(String(64), index=True)
    prompt: Mapped[list[Any]] = mapped_column(default=list)
    chosen: Mapped[list[Any]] = mapped_column(default=list)
    rejected: Mapped[list[Any]] = mapped_column(default=list)
    error_type: Mapped[str] = mapped_column(String(40), index=True)
    component: Mapped[str | None] = mapped_column(String(32), index=True)


# --- Operations ------------------------------------------------------------------------------


class LLMCall(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_deployment_ts", "deployment", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route: Mapped[str] = mapped_column(String(32), index=True)
    deployment: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tier: Mapped[str] = mapped_column(String(8))
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | rate_limited | quota | error | invalid_output
    error: Mapped[str | None] = mapped_column(Text)
    response_text: Mapped[str | None] = mapped_column(Text)  # cached output for status == ok
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class GenTask(Base):
    __tablename__ = "gen_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(40), index=True)
    key: Mapped[str] = mapped_column(String(200), unique=True)  # idempotency key, e.g. "qa:<chunk_id>"
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending/running/done/failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Release(Base):
    __tablename__ = "releases"

    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    filters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    counts: Mapped[dict[str, Any]] = mapped_column(default=dict)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
