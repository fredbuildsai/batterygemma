import pytest

from batterygemma.llm.schemas import (
    BatchClaimPairsOut,
    BatchExtractionOut,
    DPORejectionOut,
    FalsePremiseOut,
    IdeationJudgeOut,
    IdeationOut,
    QAGenerationOut,
    QAJudgeOut,
    json_validator,
)

VALID_CHUNK_EXTRACTION = {
    "chunk_index": 0,
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
VALID_EXTRACTION = {"results": [VALID_CHUNK_EXTRACTION]}


def test_valid_extraction_parses():
    result = BatchExtractionOut.model_validate(VALID_EXTRACTION)
    assert result.results[0].chunk_index == 0
    assert result.results[0].facts[0].material.name == "NMC811"
    assert result.results[0].facts[0].material.material_class == "layered oxide"  # aliased from "class"
    assert result.results[0].comparisons[0].direction == "improves"


def test_a_batch_of_several_chunks_tags_each_result_with_its_own_index():
    """The same mechanism handles 1 chunk or many - a batch response is just a longer `results` list,
    each entry independently tagged, not a different shape."""
    second = {**VALID_CHUNK_EXTRACTION, "chunk_index": 1, "facts": [], "comparisons": []}
    result = BatchExtractionOut.model_validate({"results": [VALID_CHUNK_EXTRACTION, second]})
    assert [r.chunk_index for r in result.results] == [0, 1]
    assert len(result.results[0].facts) == 1
    assert result.results[1].facts == []


def test_other_component_is_accepted_for_binder_and_conductive_carbon():
    bad = {**VALID_CHUNK_EXTRACTION["facts"][0]}
    bad["material"] = {**bad["material"], "name": "PVDF binder", "component": "other"}
    result = BatchExtractionOut.model_validate({"results": [{**VALID_CHUNK_EXTRACTION, "facts": [bad]}]})
    assert result.results[0].facts[0].material.component == "other"


def test_triple_must_have_three_elements_or_be_empty():
    bad = {**VALID_CHUNK_EXTRACTION["facts"][0], "triple": ["only", "two"]}
    with pytest.raises(ValueError, match="triple"):
        BatchExtractionOut.model_validate({"results": [{**VALID_CHUNK_EXTRACTION, "facts": [bad]}]})


def test_unknown_category_is_rejected():
    bad = {**VALID_CHUNK_EXTRACTION["facts"][0], "category": "not-a-real-category"}
    with pytest.raises(ValueError):
        BatchExtractionOut.model_validate({"results": [{**VALID_CHUNK_EXTRACTION, "facts": [bad]}]})


def test_empty_evidence_sentence_is_rejected():
    bad = {**VALID_CHUNK_EXTRACTION["facts"][0], "evidence_sentence": "   "}
    with pytest.raises(ValueError, match="empty"):
        BatchExtractionOut.model_validate({"results": [{**VALID_CHUNK_EXTRACTION, "facts": [bad]}]})


def test_facts_and_comparisons_both_default_to_empty():
    result = BatchExtractionOut.model_validate({"results": [{"chunk_index": 0}]})
    assert result.results[0].facts == [] and result.results[0].comparisons == []


def test_results_defaults_to_empty():
    result = BatchExtractionOut.model_validate({})
    assert result.results == []


VALID_CHUNK_PAIRS = {
    "chunk_index": 0,
    "pairs": [
        {"sentence_1": "Capacity fades faster above 4.4 V.", "sentence_2": "Above 4.4 V, capacity fades faster.",
         "category": "paraphrase", "subset_name": "entity"},
        {"sentence_1": "Capacity fades faster above 4.4 V.", "sentence_2": "Capacity fades slower above 4.4 V.",
         "category": "contradiction", "subset_name": "swap"},
    ],
}
VALID_PAIRS = {"results": [VALID_CHUNK_PAIRS]}


def test_valid_claim_pairs_parse():
    result = BatchClaimPairsOut.model_validate(VALID_PAIRS)
    assert len(result.results[0].pairs) == 2
    assert result.results[0].pairs[1].category == "contradiction"


class TestJsonValidator:
    def test_parses_plain_json(self):
        validate = json_validator(BatchExtractionOut)
        result = validate('{"results": [{"chunk_index": 0, "facts": [], "comparisons": []}]}')
        assert isinstance(result, BatchExtractionOut)

    def test_strips_a_markdown_fence(self):
        validate = json_validator(BatchClaimPairsOut)
        text = "Sure, here you go:\n```json\n" + '{"results": [{"chunk_index": 0, "pairs": []}]}' + "\n```"
        assert isinstance(validate(text), BatchClaimPairsOut)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            json_validator(BatchExtractionOut)("this is not json at all")

    def test_schema_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="BatchExtractionOut"):
            # missing material inside a result entry
            json_validator(BatchExtractionOut)('{"results": [{"chunk_index": 0, "facts": [{"property": "x"}]}]}')

    def test_a_garbled_object_sharing_no_fields_with_the_schema_is_rejected(self):
        """Regression test for a real incident: `openrouter-qwen3.8-27b` returned the truncated/garbled
        `{"": "results"}` for a BatchExtractionOut request. Pydantic alone accepted it silently (`results`
        defaults to `[]`), and the router then cached that permanently-empty "ok" result forever for that
        batch's prompt - this rejection is what should have made the router treat it as invalid_output and
        fail over to the next deployment instead."""
        with pytest.raises(ValueError, match="shares no fields"):
            json_validator(BatchExtractionOut)('{"": "results"}')

    def test_a_genuinely_empty_object_still_validates(self):
        """An honestly empty response (e.g. the model reporting "nothing found" via `{}`) must still be
        accepted - only content that matches none of the schema's fields is rejected, not emptiness itself."""
        result = json_validator(BatchExtractionOut)("{}")
        assert result.results == []


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


def test_json_schema_response_format_shape():
    from batterygemma.llm.schemas import json_schema_response_format

    rf = json_schema_response_format(BatchExtractionOut)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "BatchExtractionOut"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == BatchExtractionOut.model_json_schema()
