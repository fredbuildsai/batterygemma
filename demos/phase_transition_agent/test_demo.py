"""Tests for the demo (run: `pytest demos/phase_transition_agent`). Hermetic: temp database, scripted fake model."""

import sys
from pathlib import Path

import pytest
from corpusforge.models import Document
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent))

from agent import Answer, ask, make_chat  # noqa: E402
from tools import check_answer, find_facts, load_topics, plan_searches  # noqa: E402
from voice import analyse  # noqa: E402

from batterygemma.db.models import Fact  # noqa: E402
from batterygemma.db.session import get_session, init_db, register_sqlite_pragmas  # noqa: E402

QUESTION = "What happens to NMC811 cathodes when charged above 4.2 V?"
S_H2H3 = "It has been reported that the H2-H3 transition causes lattice shrinkage along the c-direction and microcracks."
S_VOLT = "When the voltage is above 4.11 V, the NCM811 electrode transforms from H2 to H3 phase with lattice oxygen oxidation."
S_NOISE = "The FePO4 precursor was prepared with H3PO4 solution."


@pytest.fixture
def engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'demo.db'}", future=True))
    init_db(engine)
    with get_session(engine) as s:
        for i, doi in enumerate(["10.1/a", "10.1/b", "10.1/c"]):
            s.add(Document(doc_id=f"d{i}", source="t", external_id=str(i), title=f"Paper {i}", norm_title=f"paper {i}",
                           doi=doi))
        for i, (sentence, doc) in enumerate([(S_H2H3, "d0"), (S_VOLT, "d1"), (S_NOISE, "d2"), (S_VOLT, "d1")]):
            s.add(Fact(id=f"f{i}", doc_ids=[doc], chunk_ids=[f"{doc}#c0"], property="p", evidence_sentence=sentence,
                       status="generated"))
        s.add(Fact(id="rej", doc_ids=["d2"], chunk_ids=["d2#c0"], property="p", status="rejected",
                   evidence_sentence="The rejected H2 H3 phase transition sentence must never be retrieved."))
    return engine


def test_the_question_fires_the_high_voltage_rule_and_names_the_material():
    plan = plan_searches(QUESTION, load_topics())
    assert plan["rules"] == ["high-voltage behaviour of layered oxides"]
    assert "h2-h3" in plan["search"] and "nmc811" in plan["materials"]
    assert "H2" in plan["must_mention"]


def test_an_unrelated_question_fires_no_rule_and_retrieves_nothing(engine):
    plan = plan_searches("What colour is graphite?", load_topics())
    assert plan["rules"] == [] and find_facts(engine, plan["search"], plan["materials"]) == []


def test_retrieval_ranks_relevant_sentences_dedupes_skips_rejected_and_attaches_the_doi(engine):
    plan = plan_searches(QUESTION, load_topics())
    evidence = find_facts(engine, plan["search"], plan["materials"])
    sentences = [e.sentence for e in evidence]
    assert S_H2H3 in sentences and S_VOLT in sentences
    assert sentences.count(S_VOLT) == 1  # the same sentence extracted twice is collapsed
    assert not any("rejected" in x for x in sentences)
    assert {e.doi for e in evidence} >= {"10.1/a", "10.1/b"}
    assert [e.number for e in evidence] == list(range(1, len(evidence) + 1))


def test_check_answer_flags_invented_citations_unsupported_claims_and_missing_topics(engine):
    plan = plan_searches(QUESTION, load_topics())
    evidence = find_facts(engine, plan["search"], plan["materials"])
    n = next(e.number for e in evidence if e.sentence == S_H2H3)

    good = check_answer(f"The H2-H3 transition causes lattice shrinkage along the c-direction and microcracks [{n}]. "
                        "Lattice oxygen is oxidised above 4.11 V as H2 turns into H3.", evidence, plan["must_mention"])
    assert good.bad_citations == [] and good.weakly_supported == []

    bad = check_answer("Graphite anodes swell during sodiation in sodium cells [99]. "
                       f"The cathode melts into a puddle of molten glass at room temperature [{n}].", evidence, ["H2"])
    assert bad.bad_citations == [99]
    assert bad.weakly_supported and "molten glass" in bad.weakly_supported[0][1]
    assert bad.missing_topics == ["H2"] and not bad.ok and "do not exist" in bad.feedback()


def test_alternatives_in_must_mention_accept_any_spelling(engine):
    ev = find_facts(engine, ["h2-h3"], [])
    assert check_answer("Shrinkage along the c-direction.", ev, ["c-axis|c-direction"]).missing_topics == []
    assert check_answer("Nothing relevant here.", ev, ["c-axis|c-direction"]).missing_topics == ["c-axis"]


def test_model_quirks_are_normalised_before_checking(engine):
    ev = find_facts(engine, ["h2-h3"], [])
    n = next(e.number for e in ev if e.sentence == S_H2H3)
    quirky = f"The H2\u2011H3 transition shrinks the c\u2011axis and causes micro-cracks in the lattice 【{n}†1】."
    verdict = check_answer(quirky, ev, ["H2", "c-axis"])
    assert verdict.bad_citations == [] and verdict.missing_topics == [] and verdict.uncited_sentences == []


def scripted(replies):
    seen = []

    def chat(messages):
        seen.append(messages)
        return replies[min(len(seen) - 1, len(replies) - 1)]
    chat.seen = seen
    return chat


def test_agent_puts_policy_and_numbered_evidence_in_the_prompt_and_returns_a_checked_answer(engine):
    chat = scripted(["The H2-H3 transition shrinks the lattice along the c-direction and causes microcracks [1]. "
                     "Lattice oxygen is oxidised above 4.11 V, where H2 becomes H3 [2]."])
    result = ask(chat, engine, QUESTION)
    system, user = chat.seen[0][0]["content"], chat.seen[0][1]["content"]
    assert "Ground every specific claim" in system and "EVIDENCE" in user and "[1]" in user and "doi:10.1/" in user
    assert QUESTION in user
    assert isinstance(result, Answer) and not result.repaired and result.verdict.ok
    assert [line.split(":")[0] for line in result.trace] == ["understand", "retrieve", "draft", "verify"]


def test_agent_repairs_once_when_the_check_fails_and_stops_there(engine):
    chat = scripted(["Phase changes occur [42].", "Still vague [42]."])
    result = ask(chat, engine, QUESTION)
    assert result.repaired and len(chat.seen) == 2  # exactly one repair round, never a loop
    feedback = chat.seen[1][-1]["content"]
    assert "do not exist" in feedback and "[42]" in feedback and "H2" in feedback
    assert result.text == "Still vague [42]." and not result.verdict.ok and result.trace[-1].startswith("repair")


def test_repair_can_be_disabled(engine):
    chat = scripted(["Vague."])
    assert not ask(chat, engine, QUESTION, repair=False).repaired and len(chat.seen) == 1


def test_model_specs_are_validated():
    with pytest.raises(SystemExit):
        make_chat("gpt:something")
    with pytest.raises(SystemExit):
        make_chat("cloud:no-such-deployment")


GENERIC = ("Here is a breakdown of what happens. Charging NMC811 above 4.2V is dangerous and causes undesirable "
           "phase changes.\n* **Safety:** a major hazard\n* **Summary:** in short, it is important to avoid this.")
SCIENTIFIC = ("The H2-H3 transition above 4.11 V shrinks the lattice along the c-axis by about 5% [1], which is attributed "
              "to delithiation of NMC811 and has been reported to induce microcracks in secondary particles [2]. "
              "Oxygen release at 4.3 V is consistent with surface reconstruction of the layered oxide [3].")


def test_voice_metrics_separate_chatty_generic_prose_from_dense_scientific_prose():
    generic, scientific = analyse(GENERIC), analyse(SCIENTIFIC)
    assert scientific.register_score > generic.register_score + 30
    assert generic.chatty_phrases >= 3 and scientific.chatty_phrases == 0
    assert generic.bullet_or_header_lines_pct > 50 and scientific.bullet_or_header_lines_pct == 0
    assert scientific.citations_per_100w > 0 and scientific.numbers_with_units_per_100w > generic.numbers_with_units_per_100w


def test_voice_metrics_are_bounded_and_handle_empty_text():
    assert 0 <= analyse(GENERIC).register_score <= 100 and 0 <= analyse(SCIENTIFIC).register_score <= 100
    assert analyse("").register_score >= 0
