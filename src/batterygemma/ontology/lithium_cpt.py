"""Lithium-ion-relevant slices of the EMMO/BattINFO ontology, rendered as short CPT text rows.

Source data (`ontology/data/*.ttl`) is a one-time snapshot of three EMMO domain-battery dependency
modules - `chemical-substance` (materials), `electrochemistry-reference` (electrodes/electrolytes),
`battery-reference` (battery/cell types) - fetched 2026-09-22 at the versions pinned in BattINFO 0.7.0's
`battinfo.ttl` (domain-battery 0.20.2, domain-electrochemistry 0.37.2, domain-chemical-substance 0.15.0).
No network access or `battinfo`/`rdflib`/`owlready2` dependency at runtime: these are static files, parsed
with a small regex-based reader tailored to how EMMO actually formats these particular files (each class as
a `###  <IRI>` block with `rdfs:subClassOf`, `skos:prefLabel`, `skos:altLabel`, and two annotation
properties this project has confirmed hold the definition and formula text - see `_DEFINITION_PROP`/
`_FORMULA_PROP`).

Deliberately narrower than "the whole ontology": each domain module also defines many classes for
chemistries/battery types outside this project's lithium-ion scope (sodium-ion, zinc-air, lead-acid, ...).
Rather than dump everything (proportionally overwhelming the ~3,000-row CPT corpus with generic ontology
definitions, most of them irrelevant to what the model will actually see) or require an exact match against
already-extracted `Fact` rows (couples this to the annotation pipeline's progress for no real benefit - see
`select_lithium_battery_types`'s docstring), each domain is filtered by structural criteria: label text for
materials, or ontology subtree membership for electrodes/electrolytes/battery types. See the three
`select_*` functions below for exactly how each filter works.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"

# Confirmed live (2026-09-22) against real classes (e.g. NMC811, GraphiteElectrode, LithiumIonBattery):
# these two EMMO annotation property IRIs hold, respectively, the human-authored definition and the
# chemical formula, when a class has them.
_DEFINITION_PROP = "EMMO_967080e5"
_FORMULA_PROP = "EMMO_b8c10b72"

# EMMO local names in these files are consistently "<word>_<uuid>", e.g.
# "substance_e877987f_3b08_4e21_8f2e_c280e6bef52f" or "battery_04a4e5a4_e6fd_43af_b1ca_4a16d5f8886c".
_LOCAL_NAME = r"[A-Za-z]+_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}"
_CLASS_HEADER = re.compile(rf"^https://[^\s]+#({_LOCAL_NAME})$")
_SUBCLASS_OF = re.compile(rf"rdfs:subClassOf :({_LOCAL_NAME})")
_PREF_LABEL = re.compile(r'skos:prefLabel "([^"]+)"')
_ALT_LABEL = re.compile(r'skos:altLabel "([^"]+)"')
_DEFINITION = re.compile(_DEFINITION_PROP + r'[^"]*"([^"]+)"')
_FORMULA = re.compile(_FORMULA_PROP + r'[^"]*"([^"]+)"')


@dataclass(frozen=True)
class OntologyClass:
    id: str
    label: str | None
    alt_label: str | None
    definition: str | None
    formula: str | None
    parent_id: str | None


def parse_ttl_classes(ttl_text: str) -> dict[str, OntologyClass]:
    """Parse one EMMO reference-module TTL file into `{local_name: OntologyClass}`.

    Each `###  <IRI>` block is one `owl:Class`; `parent_id` is only ever set when the class's
    `rdfs:subClassOf` target is a plain local class reference (`:local_name`) - a restriction expression
    (the other common `subClassOf` shape in these files) never matches `_SUBCLASS_OF` and is silently
    treated as "no parent in this file", which is exactly the "external/base concept, stop walking up"
    signal `ancestor_chain` relies on.
    """
    by_id: dict[str, OntologyClass] = {}
    for block in ttl_text.split("###  "):
        header = block.split("\n", 1)[0].strip()
        if not (m := _CLASS_HEADER.match(header)):
            continue
        cid = m.group(1)
        by_id[cid] = OntologyClass(
            id=cid,
            label=g.group(1) if (g := _PREF_LABEL.search(block)) else None,
            alt_label=g.group(1) if (g := _ALT_LABEL.search(block)) else None,
            definition=g.group(1) if (g := _DEFINITION.search(block)) else None,
            formula=g.group(1) if (g := _FORMULA.search(block)) else None,
            parent_id=g.group(1) if (g := _SUBCLASS_OF.search(block)) else None,
        )
    return by_id


def ancestor_chain(by_id: dict[str, OntologyClass], class_id: str) -> list[OntologyClass]:
    """`class_id` itself, then each ancestor in turn, stopping at the first parent not in `by_id` (an
    external/base EMMO-core concept this file doesn't define, or no parent at all)."""
    chain: list[OntologyClass] = []
    seen: set[str] = set()
    current = by_id.get(class_id)
    while current is not None and current.id not in seen:
        chain.append(current)
        seen.add(current.id)
        current = by_id.get(current.parent_id) if current.parent_id else None
    return chain


def is_descendant_of(by_id: dict[str, OntologyClass], class_id: str, ancestor_label: str) -> bool:
    """Whether `class_id` (inclusive of itself) is-a `ancestor_label`, by walking its ancestor chain."""
    return any(c.label == ancestor_label for c in ancestor_chain(by_id, class_id))


def _expand_with_ancestors(by_id: dict[str, OntologyClass], ids: set[str]) -> set[str]:
    """Every id in `ids`, plus every ancestor of each - so a kept leaf's parent classes (e.g.
    `MixedMetalOxideCompound` above a lithium oxide salt) are kept too, as supporting context for the
    "it is a kind of ..." chain `render_definition_text` builds, even though the parent's own label
    wouldn't have matched the filter on its own."""
    return {node.id for cid in ids for node in ancestor_chain(by_id, cid)}


# No word boundaries: these are CamelCase labels with no delimiters (e.g. "LithiumHexafluorophosphate",
# "LithiumNickelManganeseCobaltOxide811"), so `\blithium\b` would never match past the first word - there's
# no boundary between "m" and "H" for regex purposes, both being word characters regardless of case.
_LITHIUM_LABEL = re.compile("lithium", re.IGNORECASE)
# Common Li-ion anode materials that aren't literally named "Lithium ..." in this ontology, so the label
# filter alone would miss them.
_NON_LITHIUM_NAMED_WHITELIST = {"Graphite", "Silicon"}


def select_lithium_relevant_substances(by_id: dict[str, OntologyClass]) -> set[str]:
    """Chemical-substance classes in scope for a lithium-ion-focused CPT corpus: anything whose own
    label mentions "lithium" (covers every Li salt and Li-containing oxide cathode chemistry: LiPF6,
    NMC811, LiFePO4, ...), plus the two common non-lithium-named anode materials, plus each match's
    full ancestor chain for context (e.g. `Salt`, `MixedMetalOxideCompound`)."""
    matched = {
        cid for cid, c in by_id.items()
        if c.label and (_LITHIUM_LABEL.search(c.label) or c.label in _NON_LITHIUM_NAMED_WHITELIST)
    }
    return _expand_with_ancestors(by_id, matched)


# Structural roots for the electrochemistry domain: unlike materials, "is this Li-ion relevant" isn't a
# label question - Electrode/Electrolyte/Separator/Binder/CurrentCollector are chemistry-agnostic
# functional categories that apply to this project's cells regardless of which chemistry fills them, so
# the whole subtree under each is kept rather than filtered by label.
_ELECTROCHEMISTRY_ROOTS = ("Electrode", "Electrolyte", "Separator", "Binder", "CurrentCollector")


def select_electrode_and_electrolyte_classes(by_id: dict[str, OntologyClass]) -> set[str]:
    matched = {
        cid for cid in by_id
        if any(is_descendant_of(by_id, cid, root) for root in _ELECTROCHEMISTRY_ROOTS)
    }
    return _expand_with_ancestors(by_id, matched)


def select_lithium_battery_types(by_id: dict[str, OntologyClass]) -> set[str]:
    """Battery-domain classes in scope: everything that is-a `LithiumBattery` (covers every
    `LithiumIonXBattery` variant, `LithiumMetalBattery`, etc.), plus its ancestor chain up through
    `BatteryCell`/`Battery` for the generic "what is a cell/battery" context those provide.

    Deliberately a subtree filter, not a match against this project's already-extracted `Fact` rows:
    that would make the ontology CPT rows depend on annotation progress (facts extraction happening to
    have run, and happening to have named a battery type verbatim) for no real benefit - the whole point
    is broader, guaranteed-in-scope coverage of the taxonomy, not just whatever the extractor already
    surfaced.
    """
    matched = {cid for cid in by_id if is_descendant_of(by_id, cid, "LithiumBattery")}
    return _expand_with_ancestors(by_id, matched)


def render_definition_text(by_id: dict[str, OntologyClass], class_id: str) -> str | None:
    """One short CPT paragraph for `class_id`: its own definition (or a formula/alt-label fallback when
    no definition exists - common for chemical-substance salts), then "It is a kind of X: ..." for each
    ancestor in turn. Returns None when there's nothing at all to say (no definition, no formula, no alt
    label, and no ancestor with any of those either) - happens for a small number of purely structural
    classes with just a label."""
    chain = ancestor_chain(by_id, class_id)
    if not chain:
        return None
    sentences: list[str] = []
    for i, cls in enumerate(chain):
        label = cls.label or cls.alt_label or cls.id
        prefix = f"{label} is" if i == 0 else f"It is a kind of {label}:"
        if cls.definition:
            body = cls.definition if cls.definition.rstrip().endswith((".", ")")) else f"{cls.definition}."
            sentence = f"{prefix} {body}"
        elif cls.formula or cls.alt_label:
            details = ", ".join(d for d in (f"formula {cls.formula}" if cls.formula else None,
                                            f"also called {cls.alt_label}" if cls.alt_label and cls.alt_label != label else None) if d)
            sentence = f"{prefix} {details}." if details else None
        else:
            sentence = None
        if sentence is not None:
            sentences.append(sentence)
    return " ".join(sentences) if sentences else None


def generate_ontology_cpt_rows(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """One CPT-shaped row (`{"text": ..., "source": "ontology", "ontology_class": ..., "domain": ...}`)
    per lithium-ion-relevant class across all three domains - see the three `select_*` functions for
    exactly what's in scope in each. A class with nothing renderable (see `render_definition_text`) is
    silently skipped, not emitted as an empty/near-empty row."""
    domains = {
        "chemical-substance": (select_lithium_relevant_substances, "chemical-substance.ttl"),
        "electrochemistry": (select_electrode_and_electrolyte_classes, "electrochemistry-reference.ttl"),
        "battery": (select_lithium_battery_types, "battery-reference.ttl"),
    }
    rows: list[dict[str, Any]] = []
    for domain, (select_fn, filename) in domains.items():
        by_id = parse_ttl_classes((data_dir / filename).read_text())
        for cid in sorted(select_fn(by_id)):
            if text := render_definition_text(by_id, cid):
                cls = by_id[cid]
                rows.append({
                    "text": text, "source": "ontology", "domain": domain,
                    "ontology_class": cls.label or cls.alt_label or cid,
                })
    return rows
