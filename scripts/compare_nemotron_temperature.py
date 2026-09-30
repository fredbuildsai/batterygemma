#!/usr/bin/env python3
"""Sweep temperature for openrouter/nemotron-3-super-120b-a12b on the real extract_facts task, using the
same 5 real chunks and methodology as scripts/compare_groq_temperature.py - so results are directly
comparable to that sweep.

Temperatures tested: 0, 0.2, 0.4, 0.7, 1.0.

Run it directly:

    uv run --no-sync python scripts/compare_nemotron_temperature.py
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
from batterygemma.llm.schemas import json_schema_response_format, ExtractionOut

from compare_models import CHUNK_IDS, CLOUD_TIMEOUT, MAX_TOKENS, log, pause, print_content_comparison, print_summary, run_one
from compare_groq_models import read_env_key

OUTPUT_DIR = Path("data/eval/model_comparison")
TEMPERATURES = [0.0, 0.2, 0.4, 0.7, 1.0]
NEMOTRON_RESPONSE_FORMAT = json_schema_response_format(ExtractionOut)


def call_nemotron_at_temp(client: OpenAI, temperature: float, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=messages,
        extra_body={"reasoning": {"enabled": False}},
        response_format=NEMOTRON_RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        timeout=CLOUD_TIMEOUT,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def run_at_temperature(client: OpenAI, temperature: float, chunks: dict[str, Chunk]) -> list[dict]:
    label = f"nemotron-super-t{temperature}"
    results = []
    for cid, chunk in chunks.items():
        messages = build_extraction_messages(chunk.text)
        pause(2, f"spacing before {label}")
        results.append(run_one(label, cid, lambda m: call_nemotron_at_temp(client, temperature, m), messages))
    return results


def main() -> None:
    log(f"Temperature sweep for openrouter/nemotron-3-super: {TEMPERATURES}")
    log(f"{len(CHUNK_IDS)} real chunks, real extract_facts prompt (with the updated component/material-"
        "identity guidance), schema-constrained decoding (response_format=json_schema, strict=true)")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_env_key("OPENROUTER_API_KEY="))

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

    summary = print_summary(results, title="TEMPERATURE SWEEP SUMMARY (nemotron-super)")
    print_content_comparison(results, list(chunks.keys()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"nemotron_temp_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
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
