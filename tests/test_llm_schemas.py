import pytest

from batterygemma.llm.schemas import ClaimPairsOut, ExtractionOut, json_validator

VALID_EXTRACTION = {
    "facts": [
        {
            "material": {"name": "NMC811", "formula": "LiNi0.8Mn0.1Co0.1O2", "class": "layered oxide",
                        "component": "cathode"},
            "property": "H2->H3 transition onset", "value": "4.2", "unit": "V vs Li/Li+",
            "conditions": {"process": "charge"}, "category": "structure", "polarity": "positive",
            "triple": ["NMC811", "undergoes_above", "H2->H3 transition"],
            "evidence_sentence": "Above 4.2 V, NMC811 undergoes the H2-H3 phase transition.",
        }
    ],
    "comparisons": [
        {
            "baseline": "polycrystalline NMC811", "modification": "single-crystal NMC811",
            "metric": "capacity retention", "direction": "improves", "magnitude": "10% higher at 300 cycles",
            "conditions": {}, "component": "cathode",
            "evidence_sentence": "Single-crystal NMC811 retained 10% more capacity than polycrystalline.",
        }
    ],
}


def test_valid_extraction_parses():
    result = ExtractionOut.model_validate(VALID_EXTRACTION)
    assert result.facts[0].material.name == "NMC811"
    assert result.facts[0].material.material_class == "layered oxide"  # aliased from "class"
    assert result.comparisons[0].direction == "improves"


def test_triple_must_have_three_elements_or_be_empty():
    bad = {**VALID_EXTRACTION["facts"][0], "triple": ["only", "two"]}
    with pytest.raises(ValueError, match="triple"):
        ExtractionOut.model_validate({"facts": [bad], "comparisons": []})


def test_unknown_category_is_rejected():
    bad = {**VALID_EXTRACTION["facts"][0], "category": "not-a-real-category"}
    with pytest.raises(ValueError):
        ExtractionOut.model_validate({"facts": [bad], "comparisons": []})


def test_empty_evidence_sentence_is_rejected():
    bad = {**VALID_EXTRACTION["facts"][0], "evidence_sentence": "   "}
    with pytest.raises(ValueError, match="empty"):
        ExtractionOut.model_validate({"facts": [bad], "comparisons": []})


def test_facts_and_comparisons_both_default_to_empty():
    result = ExtractionOut.model_validate({})
    assert result.facts == [] and result.comparisons == []


VALID_PAIRS = {
    "pairs": [
        {"sentence_1": "Capacity fades faster above 4.4 V.", "sentence_2": "Above 4.4 V, capacity fades faster.",
         "category": "paraphrase", "subset_name": "entity"},
        {"sentence_1": "Capacity fades faster above 4.4 V.", "sentence_2": "Capacity fades slower above 4.4 V.",
         "category": "contradiction", "subset_name": "swap"},
    ]
}


def test_valid_claim_pairs_parse():
    result = ClaimPairsOut.model_validate(VALID_PAIRS)
    assert len(result.pairs) == 2
    assert result.pairs[1].category == "contradiction"


class TestJsonValidator:
    def test_parses_plain_json(self):
        validate = json_validator(ExtractionOut)
        result = validate('{"facts": [], "comparisons": []}')
        assert isinstance(result, ExtractionOut)

    def test_strips_a_markdown_fence(self):
        validate = json_validator(ClaimPairsOut)
        text = "Sure, here you go:\n```json\n" + '{"pairs": []}' + "\n```"
        assert isinstance(validate(text), ClaimPairsOut)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            json_validator(ExtractionOut)("this is not json at all")

    def test_schema_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="ExtractionOut"):
            json_validator(ExtractionOut)('{"facts": [{"property": "x"}], "comparisons": []}')  # missing material
