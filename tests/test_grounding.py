from batterygemma.annotate.grounding import is_grounded, overlap_ratio

SOURCE = (
    "Above 4.2 V vs Li/Li+, NMC811 undergoes the H2->H3 phase transition, accompanied by an abrupt "
    "contraction of the c lattice parameter. The resulting anisotropic strain drives intergranular cracking."
)


def test_verbatim_sentence_is_fully_grounded():
    assert overlap_ratio("NMC811 undergoes the H2->H3 phase transition.", SOURCE) == 1.0
    assert is_grounded("NMC811 undergoes the H2->H3 phase transition.", SOURCE)


def test_light_paraphrase_still_counts_as_grounded():
    paraphrase = "The H2 to H3 transition in NMC811 happens above 4.2 V."
    assert is_grounded(paraphrase, SOURCE)


def test_fabricated_evidence_unrelated_to_source_is_not_grounded():
    fabricated = "Silicon anodes expand by 300% during lithiation, fracturing the SEI repeatedly."
    assert not is_grounded(fabricated, SOURCE)
    assert overlap_ratio(fabricated, SOURCE) < 0.3


def test_empty_evidence_is_never_grounded():
    assert overlap_ratio("", SOURCE) == 0.0
    assert not is_grounded("", SOURCE)


def test_threshold_is_configurable():
    partial = "NMC811 undergoes a mysterious transformation involving silicon dendrites."
    ratio = overlap_ratio(partial, SOURCE)
    assert 0.0 < ratio < 0.6
    assert is_grounded(partial, SOURCE, threshold=ratio - 0.01)
    assert not is_grounded(partial, SOURCE, threshold=ratio + 0.01)
