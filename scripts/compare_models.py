#!/usr/bin/env python3
"""Compare Nemotron Super (OpenRouter), Gemini 3.6 Flash (direct API), and two local Ollama models
(gemma4:e4b, qwen3.5:9b-mlx) on the real extract_facts task, against the same 5 real chunks used throughout
this project's model testing.

Order: every cloud model runs its full 5-chunk pass first, then each local model runs its full 5-chunk pass,
one model at a time - not interleaved per chunk. Sequential throughout, no concurrency (see git history: an
earlier concurrent version made things strictly worse for both Gemini, whose free-tier rate limit isn't
designed for a burst of simultaneous calls, and for local Ollama, which serializes requests to one model on
one GPU anyway - "concurrent" local calls just queued behind each other and blew through their timeouts
before generation even started).

Retry: every (model, chunk) pair gets up to 3 total attempts (2 retries), 5s apart, before being declared
failed - applied uniformly to cloud and local calls, not just cloud. Nemotron's OpenRouter pool has a
confirmed intermittent empty-response glitch that a same-prompt retry reliably clears; for local calls this
is a real cost (each attempt is 100-200s+, so a persistently-failing local chunk can take ~3x as long before
being marked failed) rather than a free safety net - not hidden, see MAX_ATTEMPTS.

Reporting: a cloud-only summary (with retry counts) prints as soon as both cloud models finish, before the
(much slower) local models start - so you see how cloud did without waiting for local to finish too.

Both local models explicitly set think:false (Ollama's request field that disables reasoning outright) and
use real schema-constrained decoding (`format: <schema>`, not `format: "json"`) - confirmed via Ollama's own
docs to use grammar-based constrained decoding, not just "trained to follow". Confirmed live on this
project's real hardware: gemma4:e4b averages ~145s/call this way (5-call sample, mean of 189.5/170.2/127.3/
113.7/124.5s) - LOCAL_TIMEOUT below is set well above the slowest of those, not guessed.

Every raw response, parsed summary, and timing is saved to a timestamped JSON file under
data/eval/model_comparison/, and this script ends with two comparison reports: schema-validity rate per
model, and a content comparison (facts/comparisons counts per chunk, side by side across every model).

Run it directly:

    uv run --no-sync python scripts/compare_models.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI

from batterygemma.annotate.facts import build_extraction_messages
from batterygemma.db.session import get_session
from batterygemma.db.models import Chunk
from batterygemma.llm.schemas import ExtractionOut, json_schema_response_format, json_validator

CHUNK_IDS = [
    "openalex:W4387956213#s00-c02",
    "openalex:W4315476827#s00-c00",
    "openalex:W3042969886#s06-c00",
    "openalex:W2240909299#s00-c00",
    "openalex:W2888035576#s05-c00",
]

MAX_TOKENS = 3000
CLOUD_TIMEOUT = 60
LOCAL_TIMEOUT = 300              # slowest real observed call was 236.9s (oMLX) / 189.5s (Ollama) - real margin
CLOUD_RETRY_WAIT_SECONDS = 5
MAX_ATTEMPTS = 3  # "each chunk is tried 3 times before declaring a failure" - applied uniformly, cloud and
                  # local: for local calls (100-200s each) this means a persistently-failing chunk can take
                  # up to ~3x as long before being marked failed - a real cost, not hidden here
GEMINI_MIN_GAP_SECONDS = 6
BETWEEN_CALL_PAUSE_SECONDS = 2
OLLAMA_NUM_CTX = 6144             # matches the global value from `bg llm context-budget`

OUTPUT_DIR = Path("data/eval/model_comparison")

validate = json_validator(ExtractionOut)
EXTRACTION_SCHEMA = ExtractionOut.model_json_schema()
NEMOTRON_RESPONSE_FORMAT = json_schema_response_format(ExtractionOut)
GEMINI_RESPONSE_FORMAT = json_schema_response_format(ExtractionOut)


def log(msg: str) -> None:
    """The whole point of this script: print progress the instant it happens, not at the end."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_env_key(prefix: str) -> str:
    for line in open(".env"):
        if line.startswith(prefix) and line.strip() != prefix.rstrip():
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no non-empty {prefix} line found in .env")


def _strip_leading_junk(text: str) -> str:
    """Defensive only: drop characters before the first '{' (a stray token has been observed slipping out
    of some local thinking models before a grammar constraint engages). No-op if already clean."""
    idx = text.find("{")
    return text[idx:] if idx > 0 else text


def call_nemotron_super(client: OpenAI, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=messages,
        extra_body={"reasoning": {"enabled": False}},
        response_format=NEMOTRON_RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        timeout=CLOUD_TIMEOUT,
    )
    return resp.choices[0].message.content or ""


def call_gemini(client: OpenAI, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model="gemini-3.6-flash",
        messages=messages,
        response_format=GEMINI_RESPONSE_FORMAT,
        max_tokens=MAX_TOKENS,
        timeout=CLOUD_TIMEOUT,
    )
    return resp.choices[0].message.content or ""


def call_local(model: str, messages: list[dict[str, str]]) -> str:
    resp = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # explicit, for both local models - required for qwen3.5 (thinking model) and
                              # gemma4 (thinking model); confirmed live this does not falsely no-op on either
            "temperature": 0,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
            "format": EXTRACTION_SCHEMA,  # schema-constrained decoding, not just format: "json"
        },
        timeout=LOCAL_TIMEOUT,
    )
    resp.raise_for_status()
    return _strip_leading_junk(resp.json().get("message", {}).get("content", ""))


def run_one(label: str, chunk_id: str, call_fn, messages: list[dict[str, str]], *,
           max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Try up to `max_attempts` times (3 by default - "each chunk is tried 3 times before declaring a
    failure", applied uniformly to cloud and local calls, not just cloud) before giving up on this
    (model, chunk) pair. Waits CLOUD_RETRY_WAIT_SECONDS between attempts. `retries` in the returned dict is
    attempts - 1, i.e. 0 for a clean first-try success."""
    log(f"-> starting {label} on {chunk_id}")
    t0 = time.time()
    attempts = 0
    last_error = None
    while attempts < max_attempts:
        attempts += 1
        try:
            text = call_fn(messages)
            elapsed = round(time.time() - t0, 2)
            if not text.strip():
                raise ValueError("empty response")
            parsed = validate(text)  # raises ValueError on schema mismatch
            log(f"<- {label} on {chunk_id}: OK after {elapsed}s ({attempts} attempt(s), "
                f"{attempts - 1} retr{'y' if attempts == 2 else 'ies'}) - "
                f"{len(parsed.facts)} facts, {len(parsed.comparisons)} comparisons")
            return {"label": label, "chunk": chunk_id, "ok": True, "elapsed": elapsed, "attempts": attempts,
                    "retries": attempts - 1, "facts": len(parsed.facts), "comparisons": len(parsed.comparisons),
                    "raw_text": text}
        except Exception as e:  # noqa: BLE001 - this script's whole job is to report every failure, not hide any
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            elapsed = round(time.time() - t0, 2)
            if attempts < max_attempts:
                log(f"<- {label} on {chunk_id}: attempt {attempts}/{max_attempts} FAILED after {elapsed}s - "
                    f"{last_error} - waiting {CLOUD_RETRY_WAIT_SECONDS}s then retrying")
                time.sleep(CLOUD_RETRY_WAIT_SECONDS)
            else:
                log(f"<- {label} on {chunk_id}: FAILED after {elapsed}s ({attempts} attempt(s)) - {last_error}")
    return {"label": label, "chunk": chunk_id, "ok": False, "elapsed": round(time.time() - t0, 2),
            "attempts": attempts, "retries": attempts - 1, "reason": last_error}


def pause(seconds: float, why: str) -> None:
    log(f"   (waiting {seconds}s - {why})")
    time.sleep(seconds)


def run_cloud_model(label: str, call_fn, chunks: dict[str, Chunk], *, min_gap: float = 0.0) -> list[dict]:
    results = []
    last_call = 0.0
    for cid, chunk in chunks.items():
        messages = build_extraction_messages(chunk.text)
        since_last = time.time() - last_call
        if min_gap and since_last < min_gap:
            pause(round(min_gap - since_last, 1), f"{label} free-tier rate-limit spacing")
        else:
            pause(BETWEEN_CALL_PAUSE_SECONDS, f"spacing before {label}")
        last_call = time.time()
        results.append(run_one(label, cid, call_fn, messages))
    return results


def run_local_model(label: str, model_name: str, chunks: dict[str, Chunk]) -> list[dict]:
    results = []
    for cid, chunk in chunks.items():
        messages = build_extraction_messages(chunk.text)
        pause(BETWEEN_CALL_PAUSE_SECONDS, f"spacing before {label} (local - will take much longer than cloud)")
        results.append(run_one(label, cid, lambda m: call_local(model_name, m), messages))
    return results


def print_summary(results: list[dict], *, title: str = "SCHEMA-VALIDITY SUMMARY") -> dict:
    log("")
    log(f"=== {title} ===")
    by_label: dict[str, list[dict]] = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)

    summary = {}
    for label, rows in by_label.items():
        ok_rows = [r for r in rows if r["ok"]]
        n_ok, n_total = len(ok_rows), len(rows)
        total_facts = sum(r.get("facts", 0) for r in ok_rows)
        total_comparisons = sum(r.get("comparisons", 0) for r in ok_rows)
        avg_time = round(sum(r["elapsed"] for r in ok_rows) / n_ok, 2) if ok_rows else None
        total_retries = sum(r.get("retries", 0) for r in rows)
        chunks_needing_retry = sum(1 for r in rows if r.get("retries", 0) > 0)
        summary[label] = {"succeeded": n_ok, "total": n_total, "total_facts": total_facts,
                          "total_comparisons": total_comparisons, "avg_seconds": avg_time,
                          "total_retries": total_retries, "chunks_needing_retry": chunks_needing_retry}
        log(f"{label}: {n_ok}/{n_total} schema-valid, {total_facts} total facts, {total_comparisons} total "
            f"comparisons, avg {avg_time}s per successful call, {total_retries} total retries across "
            f"{chunks_needing_retry}/{n_total} chunk(s)")
    return summary


def print_content_comparison(results: list[dict], chunk_ids: list[str]) -> None:
    log("")
    log("=== CONTENT COMPARISON (facts/comparisons per chunk, by model) ===")
    by_chunk: dict[str, dict[str, dict]] = {cid: {} for cid in chunk_ids}
    for r in results:
        by_chunk.setdefault(r["chunk"], {})[r["label"]] = r

    labels = sorted({r["label"] for r in results})
    header = "chunk".ljust(32) + "".join(label.ljust(24) for label in labels)
    log(header)
    for cid in chunk_ids:
        row = cid.ljust(32)
        for label in labels:
            r = by_chunk.get(cid, {}).get(label)
            if r is None:
                cell = "n/a"
            elif not r["ok"]:
                cell = "FAILED"
            else:
                cell = f"{r['facts']}f/{r['comparisons']}c ({r['elapsed']}s)"
            row += cell.ljust(24)
        log(row)


def main() -> None:
    log("Starting model comparison: nemotron-super, gemini-3.6-flash, local gemma4:e4b, local qwen3.5:9b-mlx")
    log(f"{len(CHUNK_IDS)} real chunks, real extract_facts prompt, sequential, schema-constrained decoding "
        "everywhere, think:false for local models, cloud retries once on failure")
    log("Order: all cloud models complete their full pass first, then each local model, one at a time")

    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_env_key("OPENROUTER_API_KEY="))
    gemini_client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=read_env_key("GEMINI_API_KEY=")
    )

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        log("Confirmed Ollama is running locally")
    except Exception as e:
        log(f"WARNING: Ollama does not appear to be reachable ({e!r}) - local results will fail")

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
    log("--- CLOUD: nemotron-super ---")
    results += run_cloud_model("nemotron-super", lambda m: call_nemotron_super(openrouter_client, m), chunks)

    log("")
    log("--- CLOUD: gemini-3.6-flash ---")
    results += run_cloud_model("gemini-3.6-flash", lambda m: call_gemini(gemini_client, m), chunks,
                               min_gap=GEMINI_MIN_GAP_SECONDS)

    # Cloud models are done - report on them now, before spending the (much longer) time on local models,
    # rather than making everything wait until the very end to see how cloud did.
    print_summary(results, title="CLOUD SUMMARY (nemotron-super + gemini-3.6-flash)")
    print_content_comparison(results, list(chunks.keys()))

    log("")
    log("--- LOCAL: gemma4:e4b ---")
    results += run_local_model("local-gemma4-e4b", "gemma4:e4b", chunks)

    log("")
    log("--- LOCAL: qwen3.5:9b-mlx ---")
    results += run_local_model("local-qwen3.5-9b-mlx", "qwen3.5:9b-mlx", chunks)

    summary = print_summary(results, title="FINAL SUMMARY (all four models)")
    print_content_comparison(results, list(chunks.keys()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
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
