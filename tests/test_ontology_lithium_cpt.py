from batterygemma.ontology.lithium_cpt import (
    ancestor_chain,
    generate_ontology_cpt_rows,
    is_descendant_of,
    parse_ttl_classes,
    render_definition_text,
    select_electrode_and_electrolyte_classes,
    select_lithium_battery_types,
    select_lithium_relevant_substances,
)

# A tiny, hand-written TTL fixture mirroring the real files' block shape closely enough to exercise the
# parser/filters without depending on the real (large) bundled data files.
FIXTURE_TTL = """
###  https://example.org/x#thing_00000000_0000_0000_0000_000000000001
:thing_00000000_0000_0000_0000_000000000001 rdf:type owl:Class ;
                                             skos:prefLabel "ElectrochemicalComponent"@en ;
                                             EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "A component in an electrochemical system."@en .


###  https://example.org/x#thing_00000000_0000_0000_0000_000000000002
:thing_00000000_0000_0000_0000_000000000002 rdf:type owl:Class ;
                                             rdfs:subClassOf :thing_00000000_0000_0000_0000_000000000001 ;
                                             skos:prefLabel "Electrode"@en ;
                                             EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "electronically conductive part."@en .


###  https://example.org/x#thing_00000000_0000_0000_0000_000000000003
:thing_00000000_0000_0000_0000_000000000003 rdf:type owl:Class ;
                                             rdfs:subClassOf :thing_00000000_0000_0000_0000_000000000002 ;
                                             skos:prefLabel "GraphiteElectrode"@en ;
                                             EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "electrode in which the active material is graphite."@en .


###  https://example.org/x#thing_00000000_0000_0000_0000_000000000004
:thing_00000000_0000_0000_0000_000000000004 rdf:type owl:Class ;
                                             skos:prefLabel "SodiumElectrode"@en ;
                                             EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "electrode of sodium metal."@en .


###  https://example.org/x#thing_00000000_0000_0000_0000_000000000005
:thing_00000000_0000_0000_0000_000000000005 rdf:type owl:Class ;
                                             skos:altLabel "LiPF6"@en ;
                                             skos:prefLabel "LithiumHexafluorophosphate"@en ;
                                             EMMO_b8c10b72_7cc1_4e82_b4ab_728faf504919 "LiPF6" .


###  https://example.org/x#thing_00000000_0000_0000_0000_000000000006
:thing_00000000_0000_0000_0000_000000000006 rdf:type owl:Class ;
                                             skos:prefLabel "NoInfoAtAll"@en .
"""


def by_id():
    return parse_ttl_classes(FIXTURE_TTL)


def test_parse_ttl_classes_reads_label_definition_and_parent():
    classes = by_id()
    assert len(classes) == 6
    electrode = classes["thing_00000000_0000_0000_0000_000000000002"]
    assert electrode.label == "Electrode"
    assert electrode.definition == "electronically conductive part."
    assert electrode.parent_id == "thing_00000000_0000_0000_0000_000000000001"


def test_ancestor_chain_walks_up_and_stops_at_unresolved_parent():
    graphite = ancestor_chain(by_id(), "thing_00000000_0000_0000_0000_000000000003")
    assert [c.label for c in graphite] == ["GraphiteElectrode", "Electrode", "ElectrochemicalComponent"]


def test_ancestor_chain_of_unknown_id_is_empty():
    assert ancestor_chain(by_id(), "not-a-real-id") == []


def test_is_descendant_of_true_for_self_and_transitive_ancestor():
    ids = by_id()
    assert is_descendant_of(ids, "thing_00000000_0000_0000_0000_000000000003", "Electrode")
    assert is_descendant_of(ids, "thing_00000000_0000_0000_0000_000000000002", "Electrode")  # self
    assert not is_descendant_of(ids, "thing_00000000_0000_0000_0000_000000000004", "Electrode")


def test_select_electrode_and_electrolyte_classes_excludes_unrelated_subtree():
    ids = by_id()
    selected = select_electrode_and_electrolyte_classes(ids)
    assert "thing_00000000_0000_0000_0000_000000000003" in selected  # GraphiteElectrode
    assert "thing_00000000_0000_0000_0000_000000000002" in selected  # Electrode (ancestor, kept for context)
    assert "thing_00000000_0000_0000_0000_000000000004" not in selected  # SodiumElectrode: no Electrode parent link


def test_select_lithium_relevant_substances_matches_camelcase_label_with_no_word_boundary():
    """Regression test: labels are CamelCase with no delimiters (e.g. "LithiumHexafluorophosphate"), so a
    naive `\\blithium\\b` regex would never match past the first capitalized word."""
    ids = by_id()
    selected = select_lithium_relevant_substances(ids)
    assert "thing_00000000_0000_0000_0000_000000000005" in selected


def test_render_definition_text_chains_ancestors_with_is_a_kind_of():
    ids = by_id()
    text = render_definition_text(ids, "thing_00000000_0000_0000_0000_000000000003")
    assert text == (
        "GraphiteElectrode is electrode in which the active material is graphite. "
        "It is a kind of Electrode: electronically conductive part. "
        "It is a kind of ElectrochemicalComponent: A component in an electrochemical system."
    )


def test_render_definition_text_falls_back_to_formula_and_alt_label_when_no_definition():
    ids = by_id()
    text = render_definition_text(ids, "thing_00000000_0000_0000_0000_000000000005")
    assert text == "LithiumHexafluorophosphate is formula LiPF6, also called LiPF6."


def test_render_definition_text_returns_none_when_nothing_to_say():
    ids = by_id()
    assert render_definition_text(ids, "thing_00000000_0000_0000_0000_000000000006") is None


def test_select_lithium_battery_types_only_matches_actual_lithium_battery_subtree(tmp_path):
    ttl = """
###  https://example.org/x#battery_00000000_0000_0000_0000_000000000001
:battery_00000000_0000_0000_0000_000000000001 rdf:type owl:Class ;
                                               skos:prefLabel "Battery"@en ;
                                               EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "generic battery."@en .


###  https://example.org/x#battery_00000000_0000_0000_0000_000000000002
:battery_00000000_0000_0000_0000_000000000002 rdf:type owl:Class ;
                                               rdfs:subClassOf :battery_00000000_0000_0000_0000_000000000001 ;
                                               skos:prefLabel "LithiumBattery"@en ;
                                               EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "lithium battery."@en .


###  https://example.org/x#battery_00000000_0000_0000_0000_000000000003
:battery_00000000_0000_0000_0000_000000000003 rdf:type owl:Class ;
                                               rdfs:subClassOf :battery_00000000_0000_0000_0000_000000000002 ;
                                               skos:prefLabel "LithiumIonBattery"@en ;
                                               EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "li-ion battery."@en .


###  https://example.org/x#battery_00000000_0000_0000_0000_000000000004
:battery_00000000_0000_0000_0000_000000000004 rdf:type owl:Class ;
                                               rdfs:subClassOf :battery_00000000_0000_0000_0000_000000000001 ;
                                               skos:prefLabel "ZincAirBattery"@en ;
                                               EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9 "zinc air battery."@en .
"""
    ids = parse_ttl_classes(ttl)
    selected = select_lithium_battery_types(ids)
    assert selected == {
        "battery_00000000_0000_0000_0000_000000000001",  # Battery, kept as ancestor context
        "battery_00000000_0000_0000_0000_000000000002",  # LithiumBattery
        "battery_00000000_0000_0000_0000_000000000003",  # LithiumIonBattery
    }
    assert "battery_00000000_0000_0000_0000_000000000004" not in selected  # ZincAirBattery: out of scope


def test_generate_ontology_cpt_rows_against_the_real_bundled_data():
    """Smoke test against the actual bundled TTL snapshots (not the fixture above) - every row must be
    tagged, non-empty, and traceable back to a real ontology class."""
    rows = generate_ontology_cpt_rows()
    assert len(rows) > 100  # real corpus: ~245 as of 2026-09-22
    domains = {r["domain"] for r in rows}
    assert domains == {"chemical-substance", "electrochemistry", "battery"}
    for row in rows:
        assert row["source"] == "ontology"
        assert row["text"].strip()
        assert row["ontology_class"]
    labels = {r["ontology_class"] for r in rows}
    assert "LithiumHexafluorophosphate" in labels
    assert "GraphiteElectrode" in labels
    assert "LithiumIonBattery" in labels
