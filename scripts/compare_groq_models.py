#!/usr/bin/env python3
"""Compare Groq's openai/gpt-oss-120b and openai/gpt-oss-20b (both free-tier, OpenAI-compatible) against
local gemma4:e4b (the reliability-fallback winner from the earlier comparison, kept here as a baseline) on
the real extract_facts task, using the same 5 real chunks and methodology as scripts/compare_models.py.

Groq's structured-output support (`response_format={"type":"json_schema",...}`) rejects this project's real
ExtractionOut schema under `strict: true`: it requires `additionalProperties: false` and every property
listed in `required` at *every* nesting level (confirmed live 2026-09-16 with the exact error messages), and
categorically cannot support the schema's free-form `conditions: dict[str, Any]` field under strict mode at
all (an open-ended dict has no fixed property list to require) - this is the same limitation OpenAI's own
strict structured outputs have, not a Groq-specific bug. So this script uses `strict: false`: the schema is
still sent as a hint (confirmed live to produce well-formed, fully-schema-compliant JSON on both gpt-oss
models), but our own pydantic validation (`json_validator`) is what actually enforces correctness, same as
Nemotron's `response_format` was validated before schema-constrained decoding was proven for it too.

Retry: same as compare_models.py - up to 3 attempts per (model, chunk), 5s apart.

Run it directly:

    uv run --no-sync python scripts/compare_groq_models.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # compare_models.py sits next to this script, not a package

from openai import OpenAI

from batterygemma.annotate.facts import build_extraction_messages
from batterygemma.db.session import get_session
from batterygemma.db.models import Chunk
from batterygemma.llm.schemas import ExtractionOut, json_validator

# Re-use the exact same 5 real chunks used throughout this project's model testing, so results are
# directly comparable to the Nemotron/Gemini/local-Ollama numbers in data/eval/model_comparison/.
from compare_models import (  # noqa: E402
    CHUNK_IDS, CLOUD_TIMEOUT, MAX_TOKENS,
    log, pause, print_content_comparison, print_summary, run_one,
)

OUTPUT_DIR = Path("data/eval/model_comparison")
validate = json_validator(ExtractionOut)
GROQ_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "ExtractionOut", "schema": ExtractionOut.model_json_schema(), "strict": False},
}


def read_env_key(prefix: str) -> str:
    for line in open(".env"):
        if line.startswith(prefix) and line.strip() != prefix.rstrip():
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no non-empty {prefix} line found in .env")


def call_groq(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=GROQ_RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        timeout=CLOUD_TIMEOUT,
        temperature=0,  # match the rigor already applied to local calls (call_local always sets 0) - the
                        # first Groq run left this at Groq's default (~1.0), a real confound when judging
                        # schema-field completeness (triple/formula) since higher temperature can plausibly
                        # explain a model skipping optional fields, independent of prompt wording
    )
    return resp.choices[0].message.content or ""


def run_cloud_model(label: str, call_fn, chunks: dict[str, Chunk]) -> list[dict]:
    results = []
    for cid, chunk in chunks.items():
        messages = build_extraction_messages(chunk.text)
        pause(2, f"spacing before {label}")
        results.append(run_one(label, cid, call_fn, messages))
    return results


PRIOR_RESULTS_PATH = OUTPUT_DIR / "20260915T155820Z.json"


def load_prior_local_baseline(label: str = "local-gemma4-e4b") -> list[dict]:
    """Reuse gemma4:e4b's results from the earlier full comparison run instead of re-calling the local
    model - same chunks, same prompt, same schema-constrained-decoding settings, so it's a valid baseline
    without spending the ~150s/call it costs to regenerate."""
    data = json.loads(PRIOR_RESULTS_PATH.read_text())
    return [r for r in data["results"] if r["label"] == label]


def main() -> None:
    log("Starting model comparison: groq/gpt-oss-120b, groq/gpt-oss-20b, local gemma4:e4b")
    log(f"{len(CHUNK_IDS)} real chunks, real extract_facts prompt, sequential, strict:false schema hint for "
        "Groq (see module docstring - strict:true rejects this project's real schema), think:false + "
        "schema-constrained decoding for local")

    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=read_env_key("GROQ_API_KEY="))

    with get_session() as s:
        chunks = {cid: s.get(Chunk, cid) for cid in CHUNK_IDS}
    missing = [cid for cid, c in chunks.items() if c is None]
    for cid in missing:
        log(f"chunk {cid} not found in database, skipping")
    chunks = {cid: c for cid, c in chunks.items() if c is not None}
    for cid, c in chunks.items():
        log(f"  {cid} ({c.tokens} tokens)")

    results: list[dict] = []

    log("")
    log("--- CLOUD: groq/gpt-oss-120b ---")
    results += run_cloud_model("groq-gpt-oss-120b", lambda m: call_groq(groq_client, "openai/gpt-oss-120b", m), chunks)

    log("")
    log("--- CLOUD: groq/gpt-oss-20b ---")
    results += run_cloud_model("groq-gpt-oss-20b", lambda m: call_groq(groq_client, "openai/gpt-oss-20b", m), chunks)

    print_summary(results, title="GROQ CLOUD SUMMARY (gpt-oss-120b + gpt-oss-20b)")
    print_content_comparison(results, list(chunks.keys()))

    log("")
    log(f"--- LOCAL: gemma4:e4b (baseline, reused from {PRIOR_RESULTS_PATH.name} - not re-run) ---")
    results += load_prior_local_baseline()

    summary = print_summary(results, title="FINAL SUMMARY (groq gpt-oss-120b/20b + local gemma4:e4b baseline)")
    print_content_comparison(results, list(chunks.keys()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"groq_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps({"chunk_ids": list(chunks.keys()), "summary": summary, "results": results},
                                      indent=2))
    log("")
    log(f"Full results (including raw model output) saved to {output_path}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(1)
