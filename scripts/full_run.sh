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
  stage fetch fetch
  stage parse parse
  stage annotate-facts annotate facts
  stage annotate-claims annotate claims
  stage generate-qa generate qa
  stage generate-negatives generate negatives
  stage judge-qa judge qa
  stage generate-dpo generate dpo
  stage generate-ideation generate ideation
  stage judge-ideation judge ideation
  stage stats stats
  stage export export --version v0.1
  echo "=== FULL RUN DONE status=0 $(date +%T)"
} 
