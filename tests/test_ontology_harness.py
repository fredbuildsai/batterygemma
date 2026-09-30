from batterygemma.ontology.harness import (
    SYSTEM_PROMPT,
    assemble_answer,
    build_explanation_prompt,
    find_ontology_terms,
    render_classification_section,
    render_context_block,
)


def test_find_ontology_terms_against_real_bundled_data_matches_named_entities():
    q = "Classify NMC811 and explain how it relates to a lithium ion battery, and what class is a graphite electrode?"
    matches = find_ontology_terms(q)
    labels = {m.label for m in matches}
    assert labels == {"LithiumIonBattery", "GraphiteElectrode", "LithiumNickelManganeseCobaltOxide811"}


def test_short_alt_labels_only_match_whole_words_not_substrings_of_unrelated_words():
    """Regression test for a real bug: "Si" (Silicon's alt-label) matched inside "clas-si-fy" under a
    plain substring check. Word-boundary matching must not fire there."""
    q = "Please classify this material and explain its role."
    matches = find_ontology_terms(q)
    assert matches == []


def test_a_longer_match_suppresses_a_shorter_overlapping_or_ancestor_duplicating_match():
    """"graphite electrode" matches GraphiteElectrode; bare "electrode" (an ancestor of it) must not also
    be reported separately, even though the word "electrode" literally appears within the phrase."""
    q = "What is a graphite electrode?"
    matches = find_ontology_terms(q)
    assert [m.label for m in matches] == ["GraphiteElectrode"]


def test_a_question_naming_nothing_in_scope_returns_no_matches():
    assert find_ontology_terms("What is the capital of France?") == []


def test_render_context_block_is_empty_string_for_no_matches():
    assert render_context_block([]) == ""


def test_render_context_block_shows_the_full_ancestor_chain_with_definitions_in_natural_english():
    """Class names are rendered as ordinary English ("active electrode"), never the raw CamelCase
    ontology identifier ("ActiveElectrode") - the model should never see a raw identifier to echo back."""
    matches = find_ontology_terms("What is a graphite electrode?")
    block = render_context_block(matches)
    assert block.startswith("[graphite electrode]")
    assert "is a kind of: active electrode (" in block
    assert "is a kind of: electrode (" in block
    assert "is a kind of: electrochemical component (" in block
    assert "ActiveElectrode" not in block
    assert "ElectrochemicalComponent" not in block


def test_build_explanation_prompt_shape_with_matches():
    messages = build_explanation_prompt("What is a graphite electrode?")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert "already established, do not restate" in content
    assert "[graphite electrode]" in content
    assert content.rstrip().endswith("What is a graphite electrode?")
    # The classification section is never asked of the model in this prompt - it's harness-rendered.
    assert "## Ontology Classification" not in content


def test_build_explanation_prompt_still_valid_when_nothing_matches():
    messages = build_explanation_prompt("What is the capital of France?")
    content = messages[1]["content"]
    assert "No ontology context was found" in content


def test_render_classification_section_is_deterministic_from_the_retrieved_chain_in_natural_english():
    matches = find_ontology_terms("What is a graphite electrode?")
    section = render_classification_section(matches)
    assert section.startswith("## Ontology Classification")
    assert ("graphite electrode: graphite electrode is a kind of active electrode is a kind of electrode "
           "is a kind of electrochemical component") in section
    assert "GraphiteElectrode" not in section


def test_render_classification_section_natural_name_splits_digits_from_letters():
    matches = find_ontology_terms("Classify NMC811.")
    section = render_classification_section(matches)
    assert "lithium nickel manganese cobalt oxide 811" in section


def test_render_classification_section_with_no_matches_says_so():
    section = render_classification_section([])
    assert section == "## Ontology Classification\nNo ontology context was found for this question - proceed with general domain knowledge."


def test_assemble_answer_combines_deterministic_classification_with_given_explanation():
    report = assemble_answer("What is a graphite electrode?", "  Graphite electrodes intercalate lithium.  ")
    assert report.startswith("## Ontology Classification")
    assert "graphite electrode: graphite electrode is a kind of active electrode" in report
    assert report.endswith("## Mechanism / Explanation\nGraphite electrodes intercalate lithium.")
