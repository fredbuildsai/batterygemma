"""BatteryGemma's domain tables (extracted facts, generated Q&A, ...), on their OWN declarative base.

The corpus itself (`Document`, `File`, `Chunk`, `GenTask`, `Release`) belongs to corpusforge; these tables refer
to it by plain string ids (`doc_ids`, `chunk_ids`) with no foreign keys, so the two schemas migrate independently.
`LLMCall` (the router ledger) belongs to llmrouter-free. All three live in the same database.

Uses only portable SQLAlchemy types, so the same models run on SQLite now and PostgreSQL later.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


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
