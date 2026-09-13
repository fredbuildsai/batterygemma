from sqlalchemy import select

from batterygemma.db.models import Document
from batterygemma.db.session import get_session
from batterygemma.screen.screening import score_relevance, screen_documents

SCOPE = {
    "must": [["lithium-ion", "li-ion", "lithium battery", "lithium metal"]],
    "boost": {
        "cathode": ["cathode", "NMC", "LFP"],
        "interphase": ["SEI", "CEI"],
        "degradation": ["capacity fade", "degradation"],
    },
    "comparison_only": ["sodium-ion"],
}
CFG = {"scope": SCOPE, "license_allow": ["CC0", "CC-BY", "public-domain"], "license_flag": ["CC-BY-SA"]}


def test_score_uses_whole_tokens_and_tags_categories():
    rel = score_relevance(
        "Suppressing NMC cathode degradation in lithium-ion cells",
        "SEI growth and capacity fade were tracked.",
        SCOPE,
    )
    assert rel.in_scope
    assert rel.tags == ["cathode", "interphase", "degradation"]
    assert rel.score == 2 + 3  # "lithium-ion" in the title (2 points) + three categories


def test_out_of_scope_and_token_boundaries():
    assert not score_relevance("Seismic imaging of salt domes", "No batteries.", SCOPE).in_scope
    rel = score_relevance("Lithium-ion transport", "A seismic survey.", SCOPE)
    assert rel.tags == []  # "sei" must not match inside "seismic"


def test_plural_forms_match():
    rel = score_relevance("Layered oxide cathodes for lithium batteries", "Capacity fades slowly.", SCOPE)
    assert rel.in_scope
    assert rel.tags == ["cathode", "degradation"]


def test_comparison_only_title_is_flagged():
    rel = score_relevance("Sodium-ion versus lithium-ion anodes", None, SCOPE)
    assert rel.in_scope and rel.comparison_only_title


def test_unicode_dashes_and_spaces_match_ascii_terms():
    # Real OpenAlex titles from the pilot, rejected before normalization (U+2010 HYPHEN instead of "-").
    rel = score_relevance(
        "A Perspective on the Sustainability of Cathode Materials used in Lithium‐Ion Batteries", None, SCOPE
    )
    assert rel.in_scope and rel.tags == ["cathode"]

    sodium = score_relevance(
        "The Cathode Choice for Commercialization of Sodium‐Ion Batteries",
        "Compared with lithium‑ion batteries, layered oxides offer lower cost.",
        SCOPE,
    )
    assert sodium.in_scope and sodium.comparison_only_title  # comparison-only now detected too

    en_dash = score_relevance("Cathodes for lithium–ion cells at 4.2 V", None, SCOPE)
    assert en_dash.in_scope


def add(s, doc_id, title, abstract=None, license="CC-BY-4.0", **kw):
    s.add(Document(doc_id=doc_id, source="test", external_id=doc_id, title=title, norm_title=title.lower(),
                   abstract=abstract, license=license, **kw))


def test_screen_documents_decisions(engine):
    with get_session(engine) as s:
        add(s, "t:accept", "NMC cathode degradation in lithium-ion batteries", "SEI growth.")
        add(s, "t:borderline", "Lithium-ion market outlook")
        add(s, "t:nc", "NMC cathode degradation in lithium-ion batteries", license="CC-BY-NC-ND-4.0")
        add(s, "t:unknown", "LFP cathodes for lithium-ion cells", license=None)
        add(s, "t:offtopic", "Ubiquitin refolding dynamics")
        add(s, "t:sa", "SEI formation on lithium metal with NMC cathodes and capacity fade", license="CC-BY-SA-4.0")
        add(s, "t:dup", "NMC cathode degradation in lithium-ion batteries", status="duplicate", duplicate_of="t:accept")

    with get_session(engine) as s:
        counts = screen_documents(s, CFG)
    assert counts == {"accepted": 2, "borderline": 1, "rejected": 3}

    with get_session(engine) as s:
        docs = {d.doc_id: d for d in s.scalars(select(Document))}
    assert docs["t:accept"].status == "accepted"
    assert docs["t:accept"].topic_tags == ["cathode", "interphase", "degradation"]
    assert docs["t:borderline"].status_reason == "relevance:score=2"
    assert docs["t:nc"].status_reason == "license:rejected:CC-BY-NC-ND-4.0"
    assert docs["t:unknown"].status_reason == "license:unknown:None"
    assert docs["t:offtopic"].status_reason == "relevance:out_of_scope"
    assert docs["t:sa"].status == "accepted" and docs["t:sa"].license_flagged
    assert docs["t:dup"].status == "duplicate"  # never screened
