#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

stage() {
  local name="$1"; shift
  echo "=== STAGE ${name} START $(date +%T)"
  uv run --no-sync bg "$@"
  echo "=== STAGE ${name} END status=$? $(date +%T)"
}

N=15
{
  stage parse parse --limit "$N"
  stage annotate-facts annotate facts --limit "$N"
  stage annotate-claims annotate claims --limit "$N"
  stage generate-qa generate qa --limit "$N"
  stage generate-negatives generate negatives --limit "$N"
  stage judge-qa judge qa --limit "$N"
  stage generate-dpo generate dpo --limit "$N"
  stage generate-ideation generate ideation --limit "$N"
  stage judge-ideation judge ideation --limit "$N"
  stage stats stats
  stage export export --version v0.2-local-demo
  echo "=== LOCAL DEMO RUN DONE status=0 $(date +%T)"
}
