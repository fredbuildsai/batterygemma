"""Inference-time retrieval harness: find the ontology entries a question actually names, render them as
grounding context, and build a prompt that forces the model to cite that context (or admit a gap) rather
than free-associate a plausible-sounding but ungrounded classification.

This is deliberately a *retrieval* layer, separate from `lithium_cpt.py`'s *training-data* generation - the
project's earlier "no RAG" decision was about not using retrieval to build the CPT/SFT corpus itself, not a
ban on retrieval-augmented inference. Both modules read the same bundled TTL snapshots and the same
lithium-ion-relevant filters (`select_lithium_relevant_substances` etc.), so "what's in scope" is identical
in both places - this module just serves it at question time instead of baking it into weights.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from batterygemma.ontology.lithium_cpt import (
    DATA_DIR,
    OntologyClass,
    ancestor_chain,
    parse_ttl_classes,
    select_electrode_and_electrolyte_classes,
    select_lithium_battery_types,
    select_lithium_relevant_substances,
)

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist. The ontology classification for every term "
    "in this question has already been looked up and is given below as established fact - do not restate, "
    "re-derive, rename, or contradict it. Write only the technical explanation: dense, declarative "
    "scientific prose in the register of a peer-reviewed battery-materials paper's results or discussion "
    "section - specific values, mechanisms and units where relevant, ordinary paragraphs of connected "
    "sentences. Do not use headers, bullet points, numbered lists, LaTeX markup, or meta-commentary about "
    "the ontology, the question, or the task itself (no \"Relative to the ontology:\", no \"In summary\", "
    "no restating the question) - write as if this were a passage from such a paper, not a report about one. "
    "Use ordinary material and component names in your prose (e.g. 'a mixed metal oxide', 'the active "
    "electrode', 'the battery cell'), never the raw class identifiers from the context below (e.g. not "
    "'MixedMetalOxideCompound', not 'ActiveElectrode'). A material or component belongs to a *class* - it "
    "is classified as, or is a kind of, that class; it does not 'belong to the ontology' itself, since the "
    "ontology is the whole classification scheme, not a category a thing can be a member of."
)

_NO_CONTEXT_NOTE = "No ontology context was found for this question - proceed with general domain knowledge."


@dataclass(frozen=True)
class OntologyMatch:
    domain: str
    class_id: str
    label: str
    chain: list[OntologyClass]


def _camel_to_spaced(label: str) -> str:
    """"LithiumIonBattery" -> "Lithium Ion Battery" - these labels have no delimiters, so a question
    written in natural English ("lithium ion battery") needs this to line up with them at all."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", label)


def _natural_name(label: str) -> str:
    """"MixedMetalOxideCompound" -> "mixed metal oxide compound"; "LithiumNickelManganeseCobaltOxide811" ->
    "lithium nickel manganese cobalt oxide 811". Ordinary lowercase words, not a raw ontology identifier -
    used everywhere a rendered class name is shown to the model or to a person, so nothing downstream ever
    has to see (or echo back) a bare CamelCase class name like "ActiveElectrode"."""
    spaced = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", label)  # "...Oxide811" -> "...Oxide 811"
    spaced = _camel_to_spaced(spaced)
    return spaced.lower()


_DOMAIN_FILES = {
    "chemical-substance": ("chemical-substance.ttl", select_lithium_relevant_substances),
    "electrochemistry": ("electrochemistry-reference.ttl", select_electrode_and_electrolyte_classes),
    "battery": ("battery-reference.ttl", select_lithium_battery_types),
}


def _load_domain_index(data_dir: Path = DATA_DIR) -> dict[str, tuple[dict[str, OntologyClass], set[str]]]:
    """`{domain: (by_id, in_scope_ids)}` for the three bundled domains - `in_scope_ids` is exactly what
    `lithium_cpt.generate_ontology_cpt_rows` also trains on, so retrieval and training-time coverage never
    diverge silently. A domain file missing under `data_dir` (only happens with a test fixture directory
    that doesn't bother defining all three) contributes no classes rather than raising."""
    index: dict[str, tuple[dict[str, OntologyClass], set[str]]] = {}
    for domain, (filename, select_fn) in _DOMAIN_FILES.items():
        path = data_dir / filename
        by_id = parse_ttl_classes(path.read_text()) if path.exists() else {}
        index[domain] = (by_id, select_fn(by_id) if by_id else set())
    return index


def _build_search_terms(index: dict[str, tuple[dict[str, OntologyClass], set[str]]]) -> list[tuple[str, str, str]]:
    """`[(searchable_text_lowercased, domain, class_id), ...]`, longest `searchable_text` first so a more
    specific match (e.g. "graphite electrode") is tried before a shorter one that could otherwise match as
    a substring of it (e.g. "electrode") and mask it out via `_dedupe_by_span_overlap`."""
    terms: list[tuple[str, str, str]] = []
    for domain, (by_id, in_scope) in index.items():
        for cid in in_scope:
            cls = by_id[cid]
            for candidate in (cls.label, cls.alt_label):
                if not candidate:
                    continue
                terms.append((candidate.lower(), domain, cid))
                spaced = _camel_to_spaced(candidate).lower()
                if spaced != candidate.lower():
                    terms.append((spaced, domain, cid))
    terms.sort(key=lambda t: len(t[0]), reverse=True)
    return terms


def find_ontology_terms(question: str, data_dir: Path = DATA_DIR) -> list[OntologyMatch]:
    """Every in-scope ontology class (across all three domains) named in `question`, longest/most-specific
    match first, each class returned at most once even if multiple of its search forms hit (e.g. both
    "NMC811" and its spaced full name).

    Matches on whole-word boundaries only (`\\bterm\\b`), not a bare substring check - a plain `in` check
    would let a short alt-label like "Si" (Silicon) or "Li" (Lithium) fire inside an unrelated word purely
    by coincidence (confirmed live: "Si" matched inside "clas-si-fy"). A match's character span is also
    marked consumed so a shorter, ancestor-duplicating match already covered by a longer one doesn't also
    get reported (e.g. bare "Electrode" firing separately when "graphite electrode" already matched
    `GraphiteElectrode`, whose own ancestor chain already includes `Electrode`).
    """
    index = _load_domain_index(data_dir)
    terms = _build_search_terms(index)
    normalized = question.lower()

    seen: set[str] = set()
    consumed: list[tuple[int, int]] = []
    matches: list[OntologyMatch] = []
    for text, domain, cid in terms:
        if cid in seen:
            continue
        m = re.search(rf"\b{re.escape(text)}\b", normalized)
        if not m:
            continue
        span = m.span()
        if any(span[0] < end and start < span[1] for start, end in consumed):
            continue  # overlaps a longer match already accepted - skip, don't double-report
        seen.add(cid)
        consumed.append(span)
        by_id = index[domain][0]
        chain = ancestor_chain(by_id, cid)
        matches.append(OntologyMatch(domain=domain, class_id=cid, label=chain[0].label or cid, chain=chain))
    return matches


def render_context_block(matches: list[OntologyMatch]) -> str:
    """The `[Term]\\n  is-a: X (\"def\")\\n  is-a: Y (\"def\")` block injected into the prompt - empty
    string (not a header with nothing under it) when nothing in the question matched anything in scope.
    Every class name is rendered as its ordinary-English form (`_natural_name`), never the raw CamelCase
    identifier - shown to the model this way so there's no raw identifier for it to pick up and echo back
    in its prose in the first place."""
    if not matches:
        return ""
    blocks = []
    for match in matches:
        lines = [f"[{_natural_name(match.label)}]"]
        for ancestor in match.chain[1:]:
            label = ancestor.label or ancestor.alt_label or ancestor.id
            definition = ancestor.definition or ancestor.alt_label or ""
            natural = _natural_name(label)
            lines.append(f'  is a kind of: {natural} ("{definition}")' if definition else f"  is a kind of: {natural}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_classification_section(matches: list[OntologyMatch]) -> str:
    """The `## Ontology Classification` markdown block, generated entirely by the harness from the
    retrieved chains - never by the model. This is what actually makes the ontology "strictly respected":
    a model asked to *transcribe* a chain back still garbles it under paraphrase (confirmed live: reversed
    parent/child order, a material wrongly chained under a battery-type class, self-referential chains) -
    the only way to guarantee the classification matches the ontology exactly is to never let the model
    generate that text in the first place, and template it here from `OntologyMatch.chain` instead.

    Every class name in the rendered chain is its ordinary-English form (`_natural_name`), not the raw
    ontology identifier - "mixed metal oxide", not "MixedMetalOxideCompound". A thing is a kind of that
    class, not a member "belonging to" the ontology itself, hence "is a kind of" rather than "belongs to".
    """
    if not matches:
        return "## Ontology Classification\n" + _NO_CONTEXT_NOTE
    lines = ["## Ontology Classification"]
    for match in matches:
        chain_str = " is a kind of ".join(_natural_name(c.label or c.alt_label or c.id) for c in match.chain)
        direct_parent = match.chain[1] if len(match.chain) > 1 else match.chain[0]
        definition = direct_parent.definition or direct_parent.alt_label or ""
        lines.append(f'- {_natural_name(match.label)}: {chain_str}' + (f' - "{definition}"' if definition else ""))
    return "\n".join(lines)


def build_explanation_prompt(question: str, data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    """The chat-message list (`system`, `user`) asking the model for *only* the prose explanation - the
    classification section is never delegated to the model (see `render_classification_section`). The
    retrieved context is still given to the model, as grounding for the explanation's reasoning, but the
    system prompt explicitly forbids restating it as its own section.
    """
    matches = find_ontology_terms(question, data_dir)
    context = render_context_block(matches)
    context_section = (
        f"Ontology context for terms named in the question (already established, do not restate):\n{context}"
        if context else _NO_CONTEXT_NOTE
    )
    user_content = f"{context_section}\n\nQuestion:\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def assemble_answer(question: str, explanation: str, data_dir: Path = DATA_DIR) -> str:
    """The final report: the harness-templated classification section (exact, never model-generated) plus
    the model's prose explanation (from `build_explanation_prompt`) underneath. `question` is re-run
    through `find_ontology_terms` here rather than threading the matches through as a parameter, so a
    caller only has to keep track of the question string and the model's response text."""
    matches = find_ontology_terms(question, data_dir)
    classification = render_classification_section(matches)
    return f"{classification}\n\n## Mechanism / Explanation\n{explanation.strip()}"
