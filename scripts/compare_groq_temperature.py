#!/usr/bin/env python3
"""Sweep temperature for groq/gpt-oss-120b on the real extract_facts task, using the same 5 real chunks and
methodology as scripts/compare_models.py / scripts/compare_groq_models.py.

Motivation: the first Groq test (default temperature, ~1.0) got 5/5 schema-valid, 17 facts, 6 comparisons.
A temperature=0 rerun (to control for a suspected confound) unexpectedly did *worse* - 4/5 schema-valid, 11
facts, 3 comparisons, with a new failure mode (missing/invalid `category`). Since we use `strict: false` for
Groq (the schema's free-form `conditions: dict[str, Any]` field is incompatible with strict:true - see
compare_groq_models.py's docstring), there's no hard grammar constraint, so temperature can genuinely shift
how well the model follows the schema, not just wording style. This script tests more points across the
range instead of just the two extremes, to find an actual compromise rather than guessing between them.

Temperatures tested: 0, 0.2, 0.4, 0.7, 1.0 (Groq's likely default).

Run it directly:

    uv run --no-sync python scripts/compare_groq_temperature.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from batterygemma.annotate.facts import build_extraction_messages
from batterygemma.db.session import get_session
from batterygemma.db.models import Chunk

from compare_models import CHUNK_IDS, CLOUD_TIMEOUT, MAX_TOKENS, log, pause, print_content_comparison, print_summary, run_one
from compare_groq_models import GROQ_RESPONSE_FORMAT, read_env_key

OUTPUT_DIR = Path("data/eval/model_comparison")
TEMPERATURES = [0.0, 0.2, 0.4, 0.7, 1.0]


def call_groq_at_temp(client: OpenAI, model: str, temperature: float, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=GROQ_RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        timeout=CLOUD_TIMEOUT,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def run_at_temperature(client: OpenAI, temperature: float, chunks: dict[str, Chunk]) -> list[dict]:
    label = f"gpt-oss-120b-t{temperature}"
    results = []
    for cid, chunk in chunks.items():
        messages = build_extraction_messages(chunk.text)
        pause(2, f"spacing before {label}")
        results.append(run_one(label, cid, lambda m: call_groq_at_temp(client, "openai/gpt-oss-120b", temperature, m), messages))
    return results


def main() -> None:
    log(f"Temperature sweep for groq/gpt-oss-120b: {TEMPERATURES}")
    log(f"{len(CHUNK_IDS)} real chunks, real extract_facts prompt (with the updated component/material-"
        "identity guidance), strict:false schema hint")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=read_env_key("GROQ_API_KEY="))

    with get_session() as s:
        chunks = {cid: s.get(Chunk, cid) for cid in CHUNK_IDS}
    chunks = {cid: c for cid, c in chunks.items() if c is not None}
    for cid, c in chunks.items():
        log(f"  {cid} ({c.tokens} tokens)")

    results: list[dict] = []
    for temp in TEMPERATURES:
        log("")
        log(f"--- temperature={temp} ---")
        results += run_at_temperature(client, temp, chunks)

    summary = print_summary(results, title="TEMPERATURE SWEEP SUMMARY (groq/gpt-oss-120b)")
    print_content_comparison(results, list(chunks.keys()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"groq_temp_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps({"chunk_ids": list(chunks.keys()), "temperatures": TEMPERATURES,
                                       "summary": summary, "results": results}, indent=2))
    log("")
    log(f"Full results (including raw model output) saved to {output_path}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(1)
