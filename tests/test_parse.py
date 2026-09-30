from batterygemma.parse.chunk import chunk_sections, split_sentences
from batterygemma.parse.clean import clean_text, garble_ratio
from batterygemma.parse.jats import Section, classify_section, parse_jats

JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>Cracking in NMC811 cathodes</article-title></title-group>
    <abstract><p>We study <italic>intergranular</italic> cracking.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Introduction</title>
      <p>Layered oxides crack above 4.2 V [<xref ref-type="bibr" rid="b1">1</xref>,<xref ref-type="bibr" rid="b2">2</xref>], see <xref ref-type="fig" rid="f1">Fig. 1</xref>.</p>
      <fig id="f1"><label>Fig. 1</label><caption><p>In situ XRD of the (003) reflection.</p></caption>
        <graphic xlink:href="ncomms1234-f1.jpg"/></fig>
    </sec>
    <sec><title>Experimental</title>
      <sec><title>Electrochemical testing</title>
        <p>Capacity follows <disp-formula><tex-math>Q = It</tex-math></disp-formula> for constant current.</p>
        <table-wrap><label>Table 1</label><caption><p>Cycling conditions.</p></caption>
          <table><tr><th>C-rate</th><th>Cutoff</th></tr><tr><td>C/3</td><td>4.4 V</td></tr></table></table-wrap>
      </sec>
    </sec>
  </body>
  <back><ref-list><ref id="b1">Reference that must not appear.</ref></ref-list></back>
</article>"""


def test_parse_jats_sections_citations_formulas_and_captions():
    parsed = parse_jats(JATS)
    assert parsed.title == "Cracking in NMC811 cathodes"
    assert parsed.abstract == "We study intergranular cracking."
    assert [s.path for s in parsed.sections] == [["Introduction"], ["Experimental", "Electrochemical testing"]]
    assert [s.section_type for s in parsed.sections] == ["introduction", "methods"]

    intro, methods = parsed.sections
    assert intro.paragraphs == ["Layered oxides crack above 4.2 V, see Fig. 1."]
    assert intro.captions == ["Fig. 1: In situ XRD of the (003) reflection."]
    assert intro.caption_images == [["ncomms1234-f1.jpg"]]
    assert "$Q = It$" in methods.paragraphs[0]
    assert methods.captions[0].startswith("Table 1: Cycling conditions.")
    assert "C/3 | 4.4 V" in methods.captions[0]
    assert methods.caption_images == [[]]  # the table-wrap has no <graphic> of its own
    assert all("Reference that must not appear" not in p for s in parsed.sections for p in s.paragraphs)


def test_boilerplate_sections_and_superscript_citation_ranges_are_dropped():
    xml = b"""<article><front><article-meta><title-group><article-title>T</article-title></title-group>
      </article-meta></front><body>
      <sec><title>Introduction</title>
        <p>Batteries power society<sup><xref ref-type="bibr" rid="b1">1</xref>&#8211;<xref ref-type="bibr" rid="b3">3</xref></sup>.
           Cobalt is costly<sup><xref ref-type="bibr" rid="b4">4</xref>,<xref ref-type="bibr" rid="b5">5</xref></sup>, see Fig. 2.</p>
      </sec>
      <sec><title>Acknowledgements</title><p>We thank the funders.</p></sec>
      <sec sec-type="COI-statement"><title>Declarations</title><p>None.</p></sec>
      <sec><title>Author contributions</title><p>A wrote it.</p></sec>
    </body></article>"""
    parsed = parse_jats(xml)
    assert [s.path for s in parsed.sections] == [["Introduction"]]
    assert parsed.sections[0].paragraphs == ["Batteries power society. Cobalt is costly, see Fig. 2."]


def test_classify_section():
    assert classify_section(["3 Results and discussion"]) == "results"
    assert classify_section(["Supplementary notes"]) == "other"


def test_clean_text_repairs_pdf_artifacts():
    raw = "stor-\ning 16 Mb [12] and [3, 4] of data  in a package ( ) ."
    assert clean_text(raw) == "storing 16 Mb and of data in a package."


WORDS = frozenset({"affinity", "artificial", "confirm", "specific", "identify", "influence", "first", "film",
                   "field", "sulfite", "define", "the", "thin", "force", "electric", "lithium"})


def test_clean_text_removes_soft_hyphens_and_rejoins_ligatures(monkeypatch):
    monkeypatch.setattr("batterygemma.parse.clean._words", lambda: WORDS)
    raw = "Low Li + binding af fi nity of SEI fi lms and ­LiMO 2 cathodes; the fi rst cycle."
    assert clean_text(raw) == "Low Li + binding affinity of SEI films and LiMO 2 cathodes; the first cycle."


def test_ligature_rejoin_on_real_docling_splits(monkeypatch):
    # Real left-token cases observed in the parsed PDFs: fragments join leftward, whole words do not.
    monkeypatch.setattr("batterygemma.parse.clean._words", lambda: WORDS)
    raw = ("arti fi cial force fi elds con fi rm the sul fi te effect; Speci fi cally, we identi fi ed thin fi lm "
           "in fl uence under an electric fi eld, de fi ned from fi rst-principles.")
    assert clean_text(raw) == (
        "artificial force fields confirm the sulfite effect; Specifically, we identified thin film "
        "influence under an electric field, defined from first-principles."
    )


def test_ligatures_only_join_rightward_without_a_word_list(monkeypatch):
    monkeypatch.setattr("batterygemma.parse.clean._words", lambda: frozenset())
    assert clean_text("the fi rst SEI fi lms af fi nity") == "the first SEI films af finity"


def test_clean_text_normalizes_unicode_spaces():
    # Seen in real PDF text: "Tianyu\xa0Li" (no-break space) and unit spacing with narrow/thin spaces.
    assert clean_text("Tianyu Li cycled at 4.2 V and 25 °C") == "Tianyu Li cycled at 4.2 V and 25 °C"


def test_garble_ratio_flags_axis_soup():
    assert garble_ratio("0 . 9 H D ) i n / m 0 . 7 m ( n t e") > 0.5
    assert garble_ratio("The H2-H3 transition causes anisotropic strain in NMC811.") < 0.05


def test_garble_ratio_ignores_isolated_units_symbols_and_ranges():
    prose = ("Cells cycled to 4.2 V at 25 ° C lost 12 % capacity as Li + ions were consumed, "
             "consistent with J. Power Sources 196, 5109 - 5114 (2011) & later reports.")
    assert garble_ratio(prose) == 0.0


def words(text: str) -> int:
    return len(text.split())


def test_chunks_respect_sections_targets_and_overlap():
    sentence = "Cracks expose fresh surface to the electrolyte."  # 7 words
    long_paragraph = " ".join([sentence] * 12)  # 84 words, above max_tokens=60 -> split by sentence
    sections = [
        Section(path=["Results"], section_type="results", paragraphs=[long_paragraph],
                captions=["Fig. 2: SEM cross-section."]),
        Section(path=["Conclusion"], section_type="conclusion", paragraphs=["Short closing paragraph here."]),
    ]
    drafts = chunk_sections(sections, words, target_tokens=30, max_tokens=60, overlap_tokens=10)

    results = [d for d in drafts if d.section_index == 0]
    conclusion = [d for d in drafts if d.section_index == 1]
    assert len(results) >= 3 and len(conclusion) == 1
    assert results[0].overlap_prev_tokens == 0 and results[0].captions == ["Fig. 2: SEM cross-section."]
    assert all(d.overlap_prev_tokens <= 10 for d in drafts)
    assert all(d.captions == [] for d in results[1:])
    assert all(d.tokens <= 30 + 10 + 7 for d in results)  # target + overlap + one unit of slack
    assert conclusion[0].overlap_prev_tokens == 0  # overlap never crosses a section boundary
    assert split_sentences("First one. Second one.") == ["First one.", "Second one."]


def test_chunk_images_flow_from_section_captions_to_the_first_chunk_only():
    sentence = "Cracks expose fresh surface to the electrolyte."
    long_paragraph = " ".join([sentence] * 12)
    section = Section(
        path=["Results"], section_type="results", paragraphs=[long_paragraph],
        captions=["Fig. 2: SEM cross-section.", "Fig. 3: EDS map."],
        caption_images=[["ncomms-f2.jpg"], ["ncomms-f3.jpg", "ncomms-f3.jpg"]],  # dupe on purpose
    )
    drafts = chunk_sections([section], words, target_tokens=30, max_tokens=60, overlap_tokens=10)
    first, *rest = [d for d in drafts if d.section_index == 0]
    assert first.images == ["ncomms-f2.jpg", "ncomms-f3.jpg"]  # flattened, deduped
    assert all(d.images == [] for d in rest)


def test_caption_only_section_carries_its_images_too():
    section = Section(path=["Results"], section_type="results", paragraphs=[],
                      captions=["Fig. 4: TEM image."], caption_images=[["ncomms-f4.jpg"]])
    drafts = chunk_sections([section], words)
    assert drafts[0].images == ["ncomms-f4.jpg"]
