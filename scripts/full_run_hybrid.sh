#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

stage() {
  local name="$1"; shift
  echo "=== STAGE ${name} START $(date +%T)"
  uv run --no-sync bg "$@"
  echo "=== STAGE ${name} END status=$? $(date +%T)"
}

{
  stage fetch fetch --limit 385
  stage parse parse --limit 400
  stage annotate-facts annotate facts --limit 2000
  stage annotate-claims annotate claims --limit 2000
  stage generate-qa generate qa --limit 2000
  stage generate-negatives generate negatives --limit 2000
  stage judge-qa judge qa --limit 2000
  stage generate-dpo generate dpo --limit 2000
  stage generate-ideation generate ideation --limit 2000
  stage judge-ideation judge ideation --limit 2000
  stage stats stats
  stage export export --version v0.3-hybrid
  echo "=== FULL RUN DONE status=0 $(date +%T)"
}
