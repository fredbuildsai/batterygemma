import pytest

from batterygemma.llm.schemas import (
    ClaimPairsOut,
    DPORejectionOut,
    ExtractionOut,
    FalsePremiseOut,
    IdeationJudgeOut,
    IdeationOut,
    QAGenerationOut,
    QAJudgeOut,
    json_validator,
)

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


VALID_QA_ITEM = {
    "question": "Why does capacity fade accelerate above 4.2 V in NMC811?",
    "answer": "The H2->H3 transition above 4.2 V causes anisotropic strain that opens intergranular cracks.",
    "reasoning": "H2->H3 above ~4.2V -> c-axis collapse -> strain at grain boundaries -> cracks -> impedance rise.",
    "question_type": "mechanism", "answer_type": "OPEN", "component": "cathode", "chemistry": "NMC811",
}


def test_qa_generation_parses_and_defaults_to_empty():
    result = QAGenerationOut.model_validate({"items": [VALID_QA_ITEM]})
    assert result.items[0].question_type == "mechanism"
    assert QAGenerationOut.model_validate({}).items == []


def test_qa_item_rejects_unknown_question_type():
    bad = {**VALID_QA_ITEM, "question_type": "not-a-real-type"}
    with pytest.raises(ValueError):
        QAGenerationOut.model_validate({"items": [bad]})


def test_false_premise_parses():
    result = FalsePremiseOut.model_validate({
        "prompt": "Why does raising LFP's cutoff to 4.5V give much higher capacity?",
        "flawed_element": "LFP has no further redox above its plateau; extra voltage doesn't add capacity.",
        "expert_response": "That premise is false: LFP's ~170 mAh/g capacity is set by one Li per formula unit.",
        "reasoning": "LiFePO4 stores charge on a flat two-phase plateau near 3.45 V.",
    })
    assert "false" in result.expert_response.lower()


def test_false_premise_treats_null_or_empty_as_nothing_found():
    result = FalsePremiseOut.model_validate({"prompt": None, "flawed_element": "", "expert_response": None, "reasoning": None})
    assert result.prompt is None and result.flawed_element is None


def test_dpo_rejection_parses_and_rejects_unknown_error_type():
    good = DPORejectionOut.model_validate({"rejected": "HF forms at the anode.", "error_type": "wrong_mechanism"})
    assert good.error_type == "wrong_mechanism"
    with pytest.raises(ValueError):
        DPORejectionOut.model_validate({"rejected": "x", "error_type": "not_a_real_error"})


def test_ideation_parses_ideas_list():
    result = IdeationOut.model_validate({
        "problem": "Si-rich anodes fade quickly due to ~300% volume change fracturing the SEI.",
        "constraints": ["aqueous slurry compatible"],
        "reasoning": "A more elastic, LiF-rich SEI should tolerate repeated fracture better.",
        "ideas": [{
            "hypothesis": "A crosslinked PAA/CMC binder plus FEC-rich electrolyte keeps particles connected.",
            "mechanism": "Crosslinked binder network absorbs strain; FEC forms a thin, elastic SEI.",
            "risks": "Binder swelling could increase impedance.",
            "validation_experiments": ["300-cycle retention vs baseline binder"],
            "success_metrics": [">=80% retention at 300 cycles"],
        }],
    })
    assert len(result.ideas) == 1 and result.ideas[0].validation_experiments


def test_qa_judge_rejects_out_of_range_scores():
    QAJudgeOut.model_validate({"faithfulness": 5, "correctness": 4, "specificity": 3})
    with pytest.raises(ValueError):
        QAJudgeOut.model_validate({"faithfulness": 6, "correctness": 4, "specificity": 3})


def test_ideation_judge_rejects_out_of_range_scores():
    IdeationJudgeOut.model_validate({"groundedness": 5, "correctness": 4, "novelty": 3, "feasibility": 4})
    with pytest.raises(ValueError):
        IdeationJudgeOut.model_validate({"groundedness": 0, "correctness": 4, "novelty": 3, "feasibility": 4})
