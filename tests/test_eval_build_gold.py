from corpusforge.models import Document

from batterygemma.db.models import QA, Ideation, Negative
from batterygemma.db.session import get_session
from batterygemma.eval.build_gold import build_gold_set


def seed(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", split="eval"))

        # accepted closed QA on the eval-split doc - should be selected
        s.add(QA(
            id="qa-closed-eval", doc_ids=["d2"], chunk_ids=["d2#c0"], status="accepted",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="cathode", answer_type="CLOSED", closed_label="yes",
            turns=[{"role": "user", "content": "Does X happen?"}, {"role": "assistant", "content": "Yes, because..."}],
        ))
        # accepted closed QA on the train-split doc - must be excluded (would leak into training data)
        s.add(QA(
            id="qa-closed-train", doc_ids=["d1"], chunk_ids=["d1#c0"], status="accepted",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="cathode", answer_type="CLOSED", closed_label="no",
            turns=[{"role": "user", "content": "Does Y happen?"}, {"role": "assistant", "content": "No."}],
        ))
        # not accepted - must be excluded
        s.add(QA(
            id="qa-open-generated", doc_ids=["d2"], chunk_ids=["d2#c0"], status="generated",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="cathode", answer_type="OPEN",
            turns=[{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}],
        ))
        s.add(QA(
            id="qa-open-eval", doc_ids=["d2"], chunk_ids=["d2#c0"], status="accepted",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="anode", answer_type="OPEN",
            turns=[{"role": "user", "content": "Why does Z happen?"}, {"role": "assistant", "content": "Because of Q."}],
        ))
        s.add(Negative(
            id="neg-eval", doc_ids=["d2"], chunk_ids=["d2#c0"], status="generated",
            task_format="false_premise", polarity="negative", kind="false_premise",
            prompt="False premise question", flawed_element="x", expert_response="Corrected answer",
        ))
        s.add(Ideation(
            id="idea-eval", doc_ids=["d2"], chunk_ids=["d2#c0"], status="accepted",
            task_format="grounded_ideation", polarity="positive",
            problem="Si anodes fade fast.", constraints=["aqueous slurry"], reasoning="...",
            ideas=[{"hypothesis": "h"}],
        ))
        s.add(Ideation(
            id="idea-rejected", doc_ids=["d2"], chunk_ids=["d2#c0"], status="rejected",
            task_format="grounded_ideation", polarity="positive",
            problem="Rejected problem.", ideas=[],
        ))


def test_build_gold_set_only_selects_eval_split_and_accepted_rows(engine, tmp_path):
    seed(engine)
    output = tmp_path / "gold.jsonl"
    with get_session(engine) as s:
        counts = build_gold_set(s, output)

    assert counts == {"closed_qa": 1, "open_qa": 1, "negative_detection": 1, "ideation": 1}

    lines = [line for line in output.read_text().splitlines() if line.strip()]
    assert len(lines) == 4

    import json
    items = [json.loads(line) for line in lines]
    ids = {item["source_id"] for item in items}
    assert ids == {"qa-closed-eval", "qa-open-eval", "neg-eval", "idea-eval"}

    closed = next(i for i in items if i["category"] == "closed_qa")
    assert closed["prompt"] == [{"role": "user", "content": "Does X happen?"}]
    assert closed["reference_answer"] == "Yes, because..." and closed["closed_label"] == "yes"


def test_build_gold_set_respects_target_counts(engine, tmp_path):
    seed(engine)
    output = tmp_path / "gold.jsonl"
    with get_session(engine) as s:
        counts = build_gold_set(s, output, target_counts={"closed_qa": 0, "open_qa": 1, "negative_detection": 0, "ideation": 0})
    assert counts == {"open_qa": 1}
