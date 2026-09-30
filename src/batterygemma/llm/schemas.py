"""Pydantic schemas the extraction LLM's JSON output must satisfy.

These mirror `db.models.Fact` / `Comparison` / `ClaimPair` and the enums in `configs/taxonomy.yaml`. The
enums are duplicated here as `Literal`s (pydantic needs them statically) rather than loaded from YAML —
if taxonomy.yaml's fact_categories/comparison_directions/claim_pair_* lists change, update these too.
"""

import json
import re
from collections.abc import Callable
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

ModelT = TypeVar("ModelT", bound=BaseModel)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

Component = Literal["cathode", "anode", "electrolyte", "interphase", "separator_binder", "cell", "other"]
# "other": non-active-material components with no dedicated bucket above - binder and conductive carbon/
# additive are the common cases (see annotate.facts.USER_TEMPLATE for the worked examples given to the LLM).
FactCategory = Literal[
    "structure", "electrochemical", "thermal", "mechanical", "synthesis", "characterization", "degradation",
    "cost_safety",
]
Polarity = Literal["positive", "negative", "uncertain"]
ComparisonDirection = Literal["improves", "unchanged", "worsens"]
ClaimCategory = Literal["paraphrase", "contradiction"]
ClaimSubset = Literal["swap", "numeric", "mechanism", "entity"]
QuestionType = Literal[
    "mechanism", "structure_property", "trade_off", "characterization_interpretation", "synthesis_processing",
    "failure_analysis", "quantitative", "safety_cost", "comparison", "other",
]
AnswerType = Literal["OPEN", "CLOSED"]
DpoErrorType = Literal[
    "wrong_mechanism", "wrong_magnitude_or_units", "ignored_tradeoff", "fabricated_evidence", "overclaiming",
    "thermodynamically_impossible",
]


def _non_empty(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


class MaterialOut(BaseModel):
    name: str
    formula: str | None = None
    material_class: str | None = Field(None, alias="class")
    component: Component | None = None

    model_config = {"populate_by_name": True}


class FactOut(BaseModel):
    material: MaterialOut
    property: str
    value: str | None = None
    unit: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    category: FactCategory
    polarity: Polarity = "positive"
    triple: list[str] = Field(default_factory=list)
    evidence_sentence: str

    _check_property = field_validator("property", "evidence_sentence")(_non_empty)

    @field_validator("triple")
    @classmethod
    def _triple_shape(cls, v: list[str]) -> list[str]:
        if v and len(v) != 3:
            raise ValueError("triple must have exactly 3 elements [subject, relation, object] or be empty")
        return v


class ComparisonOut(BaseModel):
    baseline: str
    modification: str
    metric: str
    direction: ComparisonDirection
    magnitude: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    component: Component | None = None
    evidence_sentence: str

    _check = field_validator("baseline", "modification", "metric", "evidence_sentence")(_non_empty)


class ChunkExtractionOut(BaseModel):
    """One chunk's worth of facts/comparisons, tagged with its position in the request - the `extract`
    route always returns a `BatchExtractionOut` (a list of these), whether the request held 1 chunk or
    many: there is one extraction mechanism, not a separate single-chunk shape and a batch shape."""

    chunk_index: int
    facts: list[FactOut] = Field(default_factory=list)
    comparisons: list[ComparisonOut] = Field(default_factory=list)


class BatchExtractionOut(BaseModel):
    """Top-level shape the `extract` route must return for a request of 1 or more chunks."""

    results: list[ChunkExtractionOut] = Field(default_factory=list)


class ClaimPairOut(BaseModel):
    sentence_1: str
    sentence_2: str
    category: ClaimCategory
    subset_name: ClaimSubset

    _check = field_validator("sentence_1", "sentence_2")(_non_empty)


class ChunkClaimPairsOut(BaseModel):
    """One chunk's worth of claim pairs, tagged with its position in the request - see
    `ChunkExtractionOut` for why there's one mechanism for 1 chunk or many, not two."""

    chunk_index: int
    pairs: list[ClaimPairOut] = Field(default_factory=list)


class BatchClaimPairsOut(BaseModel):
    """Top-level shape the `extract` route must return for a claim-pairs request of 1 or more chunks."""

    results: list[ChunkClaimPairsOut] = Field(default_factory=list)


class QAItemOut(BaseModel):
    question: str
    answer: str
    reasoning: str
    question_type: QuestionType
    answer_type: AnswerType
    component: Component | None = None
    chemistry: str | None = None

    _check = field_validator("question", "answer", "reasoning")(_non_empty)


class QAGenerationOut(BaseModel):
    items: list[QAItemOut] = Field(default_factory=list)


class FalsePremiseOut(BaseModel):
    """All fields None/absent means "this excerpt has nothing specific enough to build a false premise from"."""

    prompt: str | None = None
    flawed_element: str | None = None
    expert_response: str | None = None
    reasoning: str | None = None

    @field_validator("prompt", "flawed_element", "expert_response", "reasoning")
    @classmethod
    def _empty_string_becomes_none(cls, v: str | None) -> str | None:
        return v.strip() or None if isinstance(v, str) else v


class DPORejectionOut(BaseModel):
    rejected: str
    error_type: DpoErrorType

    _check = field_validator("rejected")(_non_empty)


class IdeaOut(BaseModel):
    hypothesis: str
    mechanism: str
    risks: str
    validation_experiments: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)

    _check = field_validator("hypothesis", "mechanism", "risks")(_non_empty)


class IdeationOut(BaseModel):
    problem: str
    constraints: list[str] = Field(default_factory=list)
    reasoning: str
    ideas: list[IdeaOut] = Field(default_factory=list)

    _check = field_validator("problem", "reasoning")(_non_empty)


class QAJudgeOut(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    rationale: str = ""


class IdeationJudgeOut(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    rationale: str = ""


class OpenAnswerJudgeOut(BaseModel):
    """Stage 11 (Eval): does a model's free-form answer agree with the gold reference answer?"""

    correct: bool
    score: int = Field(ge=1, le=5)
    rationale: str = ""


class NegativeDetectionJudgeOut(BaseModel):
    """Stage 11 (Eval): did a model's answer catch and correct the planted flaw?"""

    caught_flaw: bool
    rationale: str = ""


def json_schema_response_format(model_cls: type[BaseModel], *, strict: bool = True) -> dict:
    """The OpenAI-style `response_format` value for schema-constrained decoding (as opposed to the generic
    `{"type": "json_object"}` mode). Confirmed via NVIDIA's own structured-output docs and a real live test
    against `nvidia/nemotron-3-super-120b-a12b:free` on 2026-09-15: with plain `json_object` mode this model
    regularly violated our schema (wrong enum values, missing required fields, wrong array length); with
    `json_schema` mode passing this exact schema, 5/5 real chunks came back fully schema-valid with zero
    violations. Not yet verified against every other deployment - callers should keep `json_mode=True` as
    the safe default and opt into this per route/call site once tested against that route's actual chain.

    `strict=False` (needed for Groq): Groq's `strict: true` mode requires `additionalProperties: false` and
    every property listed in `required` at *every* nesting level, and rejects a free-form `dict[str, Any]`
    field outright (confirmed live 2026-09-16/17 against ExtractionOut and QAJudgeOut) - the same limitation
    OpenAI's own strict structured outputs have. With `strict=False` the schema is still sent as a hint
    (confirmed live to produce well-formed, schema-compliant JSON on gpt-oss-120b for both extraction and
    judge schemas); our own pydantic validation (`json_validator`) is what actually enforces correctness.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": model_cls.__name__, "schema": model_cls.model_json_schema(), "strict": strict},
    }


def json_validator(model_cls: type[ModelT]) -> Callable[[str], ModelT]:
    """A `router.complete(..., validate=...)` callback: parses and validates `text` against `model_cls`.

    Tolerates a model wrapping its JSON in a ```json ... ``` fence despite `json_mode=True` (reasoning
    models sometimes do this anyway). Raises ValueError on any failure, which the router treats as
    `invalid_output` and retries on the next deployment.

    Also rejects a non-empty JSON object whose keys share nothing with `model_cls`'s own field names
    (e.g. a garbled/truncated `{"": "results"}` for a schema whose only field is `results`) - pydantic
    alone accepts this silently whenever every field has a default (`Field(default_factory=list)` etc.),
    which let a real truncated response from `openrouter-qwen3.8-27b` get cached as a valid, permanently
    empty "ok" result for one batch's prompt hash (confirmed live 2026-09-22: the router's cache then
    replayed that same garbage forever, since a cached "ok" is never re-validated against the live
    provider). A genuinely empty object (`{}`, e.g. a model correctly reporting "nothing found") is still
    accepted - only an object with content that matches none of the expected fields is rejected.
    """

    def validate(text: str) -> ModelT:
        candidate = text.strip()
        if fenced := _JSON_FENCE.search(candidate):
            candidate = fenced.group(1)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        if isinstance(data, dict) and data and not (set(data.keys()) & set(model_cls.model_fields.keys())):
            raise ValueError(
                f"JSON object shares no fields with {model_cls.__name__} (got keys {sorted(data.keys())}) - "
                "likely a garbled or truncated response"
            )
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"JSON did not match {model_cls.__name__}: {exc}") from exc

    return validate
