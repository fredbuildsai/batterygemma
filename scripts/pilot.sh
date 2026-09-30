#!/usr/bin/env bash
# M1 pilot: discover -> screen -> fetch -> parse -> stats, logged to data/logs/pilot-<timestamp>.log.
# Every stage prints "=== STAGE <name> START" and "=== STAGE <name> END status=<code>"; the run ends with
# "=== PILOT DONE status=<code>", so a watcher sees failures as well as success.
# Limits can be overridden: DISCOVER_LIMIT=1000 FETCH_LIMIT=800 PARSE_LIMIT=800 scripts/pilot.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG="data/logs/pilot-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG"

stage() {
  local name="$1"
  shift
  echo "=== STAGE ${name} START $(date +%T)"
  uv run --no-sync bg "$@"
  local status=$?
  echo "=== STAGE ${name} END status=${status} $(date +%T)"
  return "$status"
}

{
  df -h . | tail -1
  stage discover discover --source openalex --limit "${DISCOVER_LIMIT:-1000}" \
    && stage screen screen \
    && stage fetch fetch --limit "${FETCH_LIMIT:-800}" \
    && stage parse parse --limit "${PARSE_LIMIT:-800}" \
    && stage stats stats
  status=$?
  df -h . | tail -1
  echo "=== PILOT DONE status=${status} $(date +%T)"
} >"$LOG" 2>&1
