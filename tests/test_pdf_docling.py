from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData

import pytest

from batterygemma.parse.pdf_docling import (
    docling_to_parsed,
    is_author_list,
    is_front_matter,
    strip_flattened_citations,
)


@pytest.mark.parametrize(
    "text",
    [
        "Tianyu Li 1,3 · Xiao-Zi Yuan 1 · Lei Zhang 1 · Datong Song 1 · Kaiyuan Shi 2 · Christina Bock 2",
        "Aiping Wang 1,2, Sanket Kadam 3, Hong Li 4, Siqi Shi 1,2 and Yue Qi 3",
    ],
)
def test_author_lists_are_front_matter(text):
    assert is_author_list(text) and is_front_matter(text)


@pytest.mark.parametrize(
    "text",
    [
        "npj Computational Materials (2018) 4:15; doi:10.1038/s41524-018-0064-0",
        "Open Access This article is distributed under the terms of the Creative Commons Attribution 4.0 "
        "International License (http://creat iveco mmons .org/licen ses/by/4.0/), which permits unrestricted use.",
        "To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.",
    ],
)
def test_licence_and_journal_lines_are_front_matter(text):
    assert is_front_matter(text)


@pytest.mark.parametrize(
    "text",
    [
        "Li Wang and Hong Li 2 cells were cycled at C/3 for 300 cycles.",  # prose mentioning people and numbers
        "Cells cycled 25 times retained 92 % of their capacity. 25 samples were tested in total.",
        "The SEI allows Li + transport and blocks electrons (see Ref. 12 for details).",
    ],
)
def test_prose_is_not_front_matter(text):
    assert not is_front_matter(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("These modeling results generally agree with experimental observations. 81", "These modeling results generally agree with experimental observations."),
        ("67 These modeling results generally agree.", "These modeling results generally agree."),
        ("Capacity faded quickly. 12, 13 The SEI thickened.", "Capacity faded quickly. The SEI thickened."),
        ("A recent DFT study 17 showed higher potentials.", "A recent DFT study 17 showed higher potentials."),
        ("Cells cycled 25 times retained 92 % of their capacity. 25 samples were tested.", "Cells cycled 25 times retained 92 % of their capacity. 25 samples were tested."),
        ("300 cycles were run at 45 °C.", "300 cycles were run at 45 °C."),
    ],
)
def test_strip_flattened_citations_only_removes_unambiguous_cases(raw, expected):
    assert strip_flattened_citations(raw) == expected


def cell(text, row, col, header=False):
    return TableCell(text=text, start_row_offset_idx=row, end_row_offset_idx=row + 1,
                     start_col_offset_idx=col, end_col_offset_idx=col + 1, column_header=header)


def build_document() -> DoclingDocument:
    doc = DoclingDocument(name="paper")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Journal of Batteries 12 (2026) 1-10")
    doc.add_title("Suppressing cracking in NMC811")
    doc.add_heading("Abstract", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="We study intergranular cracking.")
    doc.add_heading("1 Introduction", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Layered oxides crack above 4.2 V.")
    doc.add_heading("2 Experimental", level=1)
    doc.add_heading("2.1 Electrochemical testing", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="Cells were cycled at C/3.")
    table_caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: Cycling conditions.")
    table = TableData(num_rows=2, num_cols=2, table_cells=[
        cell("C-rate", 0, 0, header=True), cell("Cutoff", 0, 1, header=True), cell("C/3", 1, 0), cell("4.4 V", 1, 1),
    ])
    doc.add_table(data=table, caption=table_caption)
    figure_caption = doc.add_text(label=DocItemLabel.CAPTION, text="Fig. 2: SEM cross-section after 300 cycles.")
    doc.add_picture(caption=figure_caption)
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="* Corresponding author.")
    doc.add_heading("Acknowledgements", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="We thank the funders.")
    doc.add_heading("References", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="[1] A. Author, J. Batteries 1 (2020) 1.")
    doc.add_heading("Appendix A", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Additional cycling data.")
    return doc


REAL_AUTHOR_LISTS = [
    # Exact strings from parsed PDFs (W2981478793, W2997099008): no-break spaces inside names.
    "Tianyu\xa0Li 1,3 · Xiao-Zi\xa0Yuan 1 · Lei\xa0Zhang 1 · Datong\xa0Song 1 · Kaiyuan\xa0Shi 2 · Christina\xa0Bock 2",
    "Jian\xa0Duan 1 · Xuan\xa0Tang 2 · Haifeng\xa0Dai 2 · Ying\xa0Yang 1 · Wangyan\xa0Wu 1 · Xuezhe\xa0Wei 2 · Yunhui\xa0Huang 1",
]


def test_real_author_lists_with_no_break_spaces_are_dropped():
    doc = DoclingDocument(name="paper")
    for author_list in REAL_AUTHOR_LISTS:
        doc.add_text(label=DocItemLabel.TEXT, text=author_list)
    doc.add_heading("1 Introduction", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Nickel-rich cathodes degrade through surface reconstruction.")

    parsed = docling_to_parsed(doc)
    assert [s.path for s in parsed.sections] == [["1 Introduction"]]
    assert parsed.sections[0].paragraphs == ["Nickel-rich cathodes degrade through surface reconstruction."]


def test_unlabeled_bibliography_and_front_matter_are_dropped():
    doc = DoclingDocument(name="paper")
    doc.add_text(label=DocItemLabel.TEXT, text="Received: 25 April 2019 / Accepted: 13 September 2019 © The Author(s) 2019")
    doc.add_heading("1 Introduction", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="SEI growth consumes lithium (Broussely et al. showed this in 2001).")
    doc.add_heading("24", level=1)  # a page number mistaken for a heading, followed by the bibliography as text
    for entry in (
        "Broussely, M. et al. Aging mechanism in Li ion cells. J. Power Sources 97-98, 13 - 21 (2001).",
        "Christensen, J. & Newman, J. A mathematical model for the SEI. J. Electrochem. Soc. 151, A1977 - A1988 (2004).",
        "Colclasure, A. M., Smith, K. A. & Kee, R. J. Modeling SEI films. Electrochim. Acta 58, 33 - 43 (2011).",
    ):
        doc.add_text(label=DocItemLabel.TEXT, text=entry)

    doc.add_heading("Compliance with Ethical Standards", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Conflict of interest The authors declare no conflicts of interest.")

    parsed = docling_to_parsed(doc)
    assert [s.path for s in parsed.sections] == [["1 Introduction"]]
    assert parsed.sections[0].paragraphs == ["SEI growth consumes lithium (Broussely et al. showed this in 2001)."]


def test_docling_document_maps_to_sections():
    parsed = docling_to_parsed(build_document())

    assert parsed.title == "Suppressing cracking in NMC811"
    assert parsed.abstract == "We study intergranular cracking."
    assert [s.path for s in parsed.sections] == [
        ["1 Introduction"],
        ["2 Experimental", "2.1 Electrochemical testing"],
        ["Appendix A"],
    ]
    assert [s.section_type for s in parsed.sections][:2] == ["introduction", "methods"]

    methods = parsed.sections[1]
    assert methods.paragraphs == ["Cells were cycled at C/3."]
    assert methods.captions[0].startswith("Table 1: Cycling conditions.")
    assert "C/3" in methods.captions[0] and "4.4 V" in methods.captions[0]
    assert methods.captions[1] == "Fig. 2: SEM cross-section after 300 cycles."

    all_text = " ".join(p for s in parsed.sections for p in s.paragraphs + s.captions)
    for dropped in ("Journal of Batteries", "Corresponding author", "thank the funders", "A. Author"):
        assert dropped not in all_text
    assert parsed.sections[2].paragraphs == ["Additional cycling data."]  # text after references resumes
