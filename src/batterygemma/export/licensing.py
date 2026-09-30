"""Public-release license gate.

Uses the same `screen.license.evaluate_license` allow/flag lists ingestion already applies
(`configs/sources.yaml`), so a document's public-release status is judged identically whether
it's being screened on the way in or checked on the way to a public release. ALLOWED is the only
status treated as safe to redistribute: FLAGGED (e.g. CC-BY-SA, which the project's own license
policy calls out as needing manual review, not automatic public release) and REJECTED/UNKNOWN
(no license, or a document that reached the DB without going through the ingestion license gate
at all - e.g. one added directly) are all "restricted" here.

This exists because a document's license can only be enforced at ingestion for documents that
actually go through ingestion. A manually-added document, or a bug that lets one slip through,
would otherwise ride along silently into a training export and from there into a public model
release - the failure mode this module exists to catch before that release happens.
"""

import json
from pathlib import Path
from typing import Any

from batterygemma.screen.license import ALLOWED, evaluate_license

LICENSE_STATUS_FILENAME = "LICENSE_STATUS.json"


def _classify_manifest(attribution: dict[str, dict[str, Any]], allow: list[str], flag: list[str]
                        ) -> dict[str, dict[str, Any]]:
    """One attribution manifest (as written by `export.unsloth_jsonl.AttributionManifest`) ->
    its restricted entries only, each tagged with the license-gate status that flagged it."""
    restricted = {}
    for key, entry in attribution.items():
        status = evaluate_license(entry["license"], allow, flag)
        if status != ALLOWED:
            restricted[key] = {**entry, "status": status}
    return restricted


def classify_export(output_dir: Path, allow: list[str], flag: list[str]) -> dict[str, Any]:
    """Scan every `*.attribution.json` in `output_dir` and summarize which source documents are
    safe for public release and which are not, per file and overall."""
    by_file: dict[str, Any] = {}
    all_restricted_doc_ids: set[str] = set()
    all_restricted_sources: dict[str, Any] = {}

    for path in sorted(output_dir.glob("*.attribution.json")):
        with open(path, encoding="utf-8") as f:
            attribution = json.load(f)
        restricted = _classify_manifest(attribution, allow, flag)
        by_file[path.name.removesuffix(".attribution.json") + ".jsonl"] = {
            "shareable": not restricted,
            "restricted_doc_ids": sorted({entry["doc_id"] for entry in restricted.values()}),
            "restricted_rows": sum(len(entry["chunks"]) for entry in restricted.values()),
        }
        for key, entry in restricted.items():
            all_restricted_doc_ids.add(entry["doc_id"])
            all_restricted_sources[key] = entry

    return {
        "shareable": not all_restricted_sources,
        "restricted_doc_ids": sorted(all_restricted_doc_ids),
        "restricted_sources": all_restricted_sources,
        "by_file": by_file,
        "checked_against": {"license_allow": allow, "license_flag": flag},
    }


def restricted_doc_ids_for_file(output_dir: Path, jsonl_name: str, allow: list[str], flag: list[str]
                                 ) -> set[str]:
    """Which doc_ids feeding `jsonl_name` (e.g. "cpt_train.jsonl") are not safe to redistribute."""
    attribution_path = output_dir / (jsonl_name.removesuffix(".jsonl") + ".attribution.json")
    if not attribution_path.exists():
        return set()
    with open(attribution_path, encoding="utf-8") as f:
        attribution = json.load(f)
    restricted = _classify_manifest(attribution, allow, flag)
    return {entry["doc_id"] for entry in restricted.values()}


def restricted_row_indices_for_file(output_dir: Path, jsonl_name: str, restricted_doc_ids: set[str]) -> set[int]:
    """Which row indices (0-based line numbers) of `jsonl_name` came from `restricted_doc_ids`."""
    attribution_path = output_dir / (jsonl_name.removesuffix(".jsonl") + ".attribution.json")
    if not attribution_path.exists() or not restricted_doc_ids:
        return set()
    with open(attribution_path, encoding="utf-8") as f:
        attribution = json.load(f)
    indices: set[int] = set()
    for entry in attribution.values():
        if entry["doc_id"] in restricted_doc_ids:
            indices.update(entry["chunks"])
    return indices


def write_filtered_jsonl(source: Path, drop_row_indices: set[int], destination: Path) -> int:
    """Copy `source` to `destination`, dropping the given 0-based line indices. Returns rows kept."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(source, encoding="utf-8") as fin, open(destination, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in drop_row_indices:
                continue
            fout.write(line)
            kept += 1
    return kept


def write_license_status(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_license_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
