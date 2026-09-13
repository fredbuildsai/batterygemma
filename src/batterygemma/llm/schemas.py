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

Component = Literal["cathode", "anode", "electrolyte", "interphase", "separator_binder", "cell"]
FactCategory = Literal[
    "structure", "electrochemical", "thermal", "mechanical", "synthesis", "characterization", "degradation",
    "cost_safety",
]
Polarity = Literal["positive", "negative", "uncertain"]
ComparisonDirection = Literal["improves", "unchanged", "worsens"]
ClaimCategory = Literal["paraphrase", "contradiction"]
ClaimSubset = Literal["swap", "numeric", "mechanism", "entity"]


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


class ExtractionOut(BaseModel):
    """Top-level shape the `extract` route must return for one chunk."""

    facts: list[FactOut] = Field(default_factory=list)
    comparisons: list[ComparisonOut] = Field(default_factory=list)


class ClaimPairOut(BaseModel):
    sentence_1: str
    sentence_2: str
    category: ClaimCategory
    subset_name: ClaimSubset

    _check = field_validator("sentence_1", "sentence_2")(_non_empty)


class ClaimPairsOut(BaseModel):
    pairs: list[ClaimPairOut] = Field(default_factory=list)


def json_validator(model_cls: type[ModelT]) -> Callable[[str], ModelT]:
    """A `router.complete(..., validate=...)` callback: parses and validates `text` against `model_cls`.

    Tolerates a model wrapping its JSON in a ```json ... ``` fence despite `json_mode=True` (reasoning
    models sometimes do this anyway). Raises ValueError on any failure, which the router treats as
    `invalid_output` and retries on the next deployment.
    """

    def validate(text: str) -> ModelT:
        candidate = text.strip()
        if fenced := _JSON_FENCE.search(candidate):
            candidate = fenced.group(1)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"JSON did not match {model_cls.__name__}: {exc}") from exc

    return validate
