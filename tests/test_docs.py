"""Documentation is checked like code: arc42 completeness, working links, and real test references."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARCH = ROOT / "docs" / "architecture"

ARC42_SECTIONS = [
    "01-introduction-and-goals", "02-constraints", "03-context-and-scope", "04-solution-strategy",
    "05-building-block-view", "06-runtime-view", "07-deployment-view", "08-crosscutting-concepts",
    "09-architecture-decisions", "10-quality-requirements", "11-risks-and-technical-debt", "12-glossary",
]


def markdown_files():
    return [ROOT / "README.md", *ARCH.glob("*.md"), *(ROOT / "docs").glob("*.md")]


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).replace(" ", "-")


def test_all_twelve_arc42_sections_exist_and_are_not_stubs():
    for section in ARC42_SECTIONS:
        text = (ARCH / f"{section}.md").read_text()
        assert text.startswith("# "), section
        assert len(text.split()) > 60, f"{section} looks like a stub"


def test_architecture_index_links_every_section():
    index = (ARCH / "README.md").read_text()
    for section in ARC42_SECTIONS:
        assert f"({section}.md)" in index
    assert "(appendices.md)" in index


def test_relative_markdown_links_resolve_including_anchors():
    for md in markdown_files():
        for target, anchor in re.findall(r"\]\(([^)#\s]+\.md)(?:#([^)]*))?\)", md.read_text()):
            if target.startswith("http"):
                continue
            path = md.parent / target
            assert path.exists(), f"{md.name} links to missing {target}"
            if anchor:
                headings = {slug(h) for h in re.findall(r"^#{1,6} (.*)$", path.read_text(), flags=re.MULTILINE)}
                assert anchor in headings, f"{md.name} links to {target}#{anchor}, which is not a heading there"


def test_intra_document_anchors_resolve():
    for md in markdown_files():
        text = md.read_text()
        headings = {slug(h) for h in re.findall(r"^#{1,6} (.*)$", text, flags=re.MULTILINE)}
        for anchor in re.findall(r"\]\(#([^)]+)\)", text):
            assert anchor in headings, f"{md.name} has a dead anchor #{anchor}"


def test_decisions_documenting_the_split_exist():
    adr = (ARCH / "09-architecture-decisions.md").read_text()
    for heading in ("ADR-013: Split into three packages", "ADR-014:", "ADR-015:"):
        assert heading in adr


def test_readme_documents_installing_from_github_and_the_setup_script():
    readme = (ROOT / "README.md").read_text()
    assert "git+https://github.com/fredbuildsai/llmrouter-free.git" in readme
    assert "git+https://github.com/fredbuildsai/corpusforge.git" in readme
    assert "scripts/setup.sh" in readme


def test_pyproject_pins_both_packages_to_github_tags():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "llmrouter-free @ git+https://github.com/fredbuildsai/llmrouter-free.git@v" in pyproject
    assert "corpusforge @ git+https://github.com/fredbuildsai/corpusforge.git@v" in pyproject
