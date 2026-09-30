#!/usr/bin/env bash
# Live end-to-end smoke run of stages 7-9 on a small real batch, logged with per-stage markers.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG="data/logs/pipeline-smoke-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG"

stage() {
  local name="$1"; shift
  echo "=== STAGE ${name} START $(date +%T)"
  uv run --no-sync bg "$@"
  echo "=== STAGE ${name} END status=$? $(date +%T)"
}

N="${N:-10}"
{
  stage annotate-facts annotate facts --limit "$N"
  stage annotate-claims annotate claims --limit "$N"
  stage generate-qa generate qa --limit "$N"
  stage generate-negatives generate negatives --limit "$N"
  stage judge-qa judge qa --limit "$N"
  stage generate-dpo generate dpo --limit "$N"
  stage generate-ideation generate ideation --limit "$N"
  stage judge-ideation judge ideation --limit "$N"
  stage stats stats
  stage export export --version v0.1-smoke
  echo "=== PIPELINE SMOKE DONE status=0 $(date +%T)"
} >"$LOG" 2>&1
