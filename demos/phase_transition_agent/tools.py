"""The three tools of the demo agent. Plain functions over batterygemma's own database and utilities.

Nothing here writes to the database or modifies the batterygemma package.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from corpusforge.annotate.grounding import overlap_ratio
from corpusforge.models import Document
from sqlalchemy import Engine, or_, select
from sqlalchemy.orm import Session

from batterygemma.db.models import Fact

HERE = Path(__file__).parent


def load_topics(path: Path = HERE / "topics.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Evidence:
    number: int
    sentence: str
    doi: str | None
    title: str
    score: float

    def render(self) -> str:
        return f"[{self.number}] {self.sentence.strip()} (doi:{self.doi or 'n/a'})"


# --- tool 1: understand the question ------------------------------------------------------------------------


def plan_searches(question: str, topics: dict) -> dict:
    """Which rules fire for this question, the search terms they contribute, and the materials named."""
    q = question.lower()
    fired = [r for r in topics["rules"] if any(t.lower() in q for t in r["triggers"])]
    search = list(dict.fromkeys(term.lower() for r in fired for term in r["search"]))
    material_terms: list[str] = []
    for phrase, aliases in topics["materials"].items():
        if phrase in q:
            material_terms += [a.lower() for a in aliases]
    must = list(dict.fromkeys(m for r in fired for m in r["must_mention"]))
    return {"rules": [r["name"] for r in fired], "search": search,
            "materials": list(dict.fromkeys(material_terms)), "must_mention": must}


# --- tool 2: find evidence in the extracted facts --------------------------------------------------------------


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\-\.]+", text.lower()))


def find_facts(engine: Engine, search: list[str], materials: list[str], k: int = 8) -> list[Evidence]:
    """Rank non-rejected facts by how many search terms (and material aliases) their evidence sentence contains.

    Keyword search on purpose: transparent and dependency-free. Near-duplicate sentences (the same sentence is often
    extracted as several facts) are collapsed, and at most two sentences per paper are kept so one review article
    cannot fill the whole context.
    """
    if not search:
        return []
    with Session(engine) as s:
        clauses = [Fact.evidence_sentence.ilike(f"%{t}%") for t in search]
        rows = s.execute(
            select(Fact.evidence_sentence, Fact.doc_ids).where(Fact.status != "rejected", or_(*clauses))
        ).all()
        scored: dict[str, tuple[float, str, str | None]] = {}
        for sentence, doc_ids in rows:
            low = sentence.lower()
            hits = sum(1 for t in search if t in low)
            mat = sum(1 for m in materials if m in low)
            score = hits + 0.5 * min(mat, 2) + (0.5 if re.search(r"\d", sentence) else 0)
            key = re.sub(r"\W+", " ", low)[:120]
            if key not in scored or score > scored[key][0]:
                scored[key] = (score, sentence, (doc_ids or [None])[0])
        ranked = sorted(scored.values(), key=lambda x: -x[0])
        picked: list[Evidence] = []
        per_doc: dict[str | None, int] = {}
        for score, sentence, doc_id in ranked:
            if per_doc.get(doc_id, 0) >= 2 or len(sentence) > 600:
                continue
            doc = s.get(Document, doc_id) if doc_id else None
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            picked.append(Evidence(len(picked) + 1, sentence, doc.doi if doc else None, doc.title if doc else "", score))
            if len(picked) == k:
                break
    return picked


# --- tool 3: check a draft against the evidence -------------------------------------------------------------


@dataclass
class Verdict:
    bad_citations: list[int]
    uncited_sentences: list[str]
    weakly_supported: list[tuple[int, str]]
    missing_topics: list[str]

    @property
    def ok(self) -> bool:
        return not (self.bad_citations or self.weakly_supported or self.missing_topics)

    def feedback(self) -> str:
        lines = []
        if self.bad_citations:
            lines.append(f"- You cited evidence numbers that do not exist: {self.bad_citations}. Use only the given numbers.")
        for n, sentence in self.weakly_supported:
            lines.append(f"- This sentence cites [{n}] but is not supported by it: \"{sentence}\". Fix it or remove the citation.")
        if self.missing_topics:
            lines.append("- The evidence covers these specifics but your answer omits them; name them explicitly: "
                         + ", ".join(self.missing_topics) + ".")
        return "\n".join(lines)


_CITE = re.compile(r"\[(\d+)\]")


def normalise(text: str) -> str:
    """Make model quirks comparable: Unicode hyphens/dashes -> '-', and citation styles such as 【3†1】 or (3) -> [3]."""
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2212]", "-", text)
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)


def check_answer(answer: str, evidence: list[Evidence], must_mention: list[str], threshold: float = 0.3) -> Verdict:
    answer = normalise(answer)
    by_number = {e.number: e for e in evidence}
    bad: set[int] = set()
    weak: list[tuple[int, str]] = []
    uncited: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        cites = [int(n) for n in _CITE.findall(sentence)]
        if not cites:
            uncited.append(sentence)
            continue
        plain = _CITE.sub("", sentence)
        for n in cites:
            if n not in by_number:
                bad.add(n)
            elif overlap_ratio(plain, by_number[n].sentence) < threshold and not any(
                    overlap_ratio(plain, by_number[m].sentence) >= threshold for m in cites if m in by_number):
                weak.append((n, sentence))
    evidence_text = " ".join(e.sentence for e in evidence).lower()
    answer_low = answer.lower()
    # A must_mention entry may list alternatives separated by "|" (e.g. "c-axis|c-direction"): the topic counts as
    # present in the evidence / covered by the answer if ANY alternative appears.
    missing = [t.split("|")[0] for t in must_mention
               if any(a in evidence_text for a in t.lower().split("|"))
               and not any(a in answer_low for a in t.lower().split("|"))]
    return Verdict(sorted(bad), uncited, weak, missing)


# --- tool 4 (optional): ontology classification, reusing batterygemma's harness ---------------------------------


def ontology_lookup(question: str) -> str:
    from batterygemma.ontology.harness import find_ontology_terms, render_classification_section

    return render_classification_section(find_ontology_terms(question))
