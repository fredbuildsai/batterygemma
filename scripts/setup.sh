#!/usr/bin/env bash
# One-command setup for BatteryGemma: virtualenv, the two packages it is built on (llmrouter-free and corpusforge,
# installed from GitHub), BatteryGemma itself, and a scaffolded project (configs, .env, data directories, database).
#
#   scripts/setup.sh                 # normal install, packages from their pinned GitHub tags
#   scripts/setup.sh --dev           # clone ../llmrouter-free and ../corpusforge and install all three editable
#   scripts/setup.sh --with-train    # also install the Unsloth training extra
#   scripts/setup.sh --dry-run       # print what would be done; change nothing
#
# Safe to re-run: existing environments, clones and project files are reused, never overwritten.

set -euo pipefail

LLMROUTER_REPO="${LLMROUTER_REPO:-https://github.com/fredbuildsai/llmrouter-free.git}"
CORPUSFORGE_REPO="${CORPUSFORGE_REPO:-https://github.com/fredbuildsai/corpusforge.git}"
LLMROUTER_REF="${LLMROUTER_REF:-v0.1.0}"
CORPUSFORGE_REF="${CORPUSFORGE_REF:-v0.1.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"

DEV=0
WITH_TRAIN=0
WITH_PARSE=1
SKIP_INIT=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: scripts/setup.sh [options]

Options:
  --dev            Clone llmrouter-free and corpusforge next to this repository (if missing) and install all three
                   as editable installs, for working across repositories.
  --with-train     Also install the Unsloth training extra (Apple Silicon MLX backend; large download).
  --no-parse       Skip the parsing extras (Docling pulls PyTorch; needed for 'bg parse' only).
  --skip-init      Do not run 'bg init' (configs, .env, data directories, database) at the end.
  --python VER     Python version for the virtualenv (default: ${PYTHON_VERSION}; supported: 3.11, 3.12).
  --dry-run        Print every command instead of running it.
  -h, --help       Show this help.

Environment overrides: LLMROUTER_REF / CORPUSFORGE_REF (git tag or branch to install; default ${LLMROUTER_REF}),
LLMROUTER_REPO / CORPUSFORGE_REPO (git URLs).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dev) DEV=1 ;;
    --with-train) WITH_TRAIN=1 ;;
    --no-parse) WITH_PARSE=0 ;;
    --skip-init) SKIP_INIT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --python)
      [ $# -ge 2 ] || { echo "error: --python needs a version" >&2; exit 2; }
      PYTHON_VERSION="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$PYTHON_VERSION" in
  3.11|3.12) ;;
  *) echo "error: unsupported Python '$PYTHON_VERSION' (BatteryGemma supports 3.11 and 3.12)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
cd "$ROOT"

step() { printf '\n==> %s\n' "$*"; }

# Run a command, or only print it under --dry-run.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# ---- 1. prerequisites --------------------------------------------------------------------------------------
step "Checking prerequisites"
if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required (https://git-scm.com/downloads)" >&2
  exit 1
fi
echo "git: $(command -v git)"

if command -v uv >/dev/null 2>&1; then
  echo "uv: $(command -v uv)"
else
  echo "uv not found - installing it with the official installer ($UV_INSTALLER_URL)"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '+ curl -LsSf %s | sh\n' "$UV_INSTALLER_URL"
    # shellcheck disable=SC2016  # literal text for the dry-run display
    printf '+ export PATH="$HOME/.local/bin:$PATH"\n'
  else
    command -v curl >/dev/null 2>&1 || { echo "error: curl is required to install uv" >&2; exit 1; }
    curl -LsSf "$UV_INSTALLER_URL" | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "error: uv installed but not on PATH; open a new shell and re-run" >&2; exit 1; }
  fi
fi

# ---- 2. virtualenv -----------------------------------------------------------------------------------------
step "Creating the virtual environment (.venv, Python $PYTHON_VERSION)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  echo ".venv already exists - reusing it"
else
  run uv venv "$ROOT/.venv" --python "$PYTHON_VERSION"
fi
export VIRTUAL_ENV="$ROOT/.venv"

# ---- 3. packages -------------------------------------------------------------------------------------------
BG_EXTRAS="dev"
[ "$WITH_PARSE" -eq 1 ] && BG_EXTRAS="$BG_EXTRAS,parse,dedupe"
[ "$WITH_TRAIN" -eq 1 ] && BG_EXTRAS="$BG_EXTRAS,train"
CF_SPEC="corpusforge"
[ "$WITH_PARSE" -eq 1 ] && CF_SPEC="corpusforge[parse]"

if [ "$DEV" -eq 1 ]; then
  step "Development mode: sibling checkouts, all three packages editable"
  for pair in "llmrouter-free|$LLMROUTER_REPO|$LLMROUTER_REF" "corpusforge|$CORPUSFORGE_REPO|$CORPUSFORGE_REF"; do
    name="${pair%%|*}"; rest="${pair#*|}"; repo="${rest%%|*}"; ref="${rest#*|}"
    if [ -d "$PARENT/$name/.git" ]; then
      echo "$PARENT/$name already cloned - leaving it as is"
    else
      run git clone "$repo" "$PARENT/$name"
      run git -C "$PARENT/$name" checkout "$ref"
    fi
  done
  run uv pip install -e "$PARENT/llmrouter-free" -e "$PARENT/$CF_SPEC" -e ".[$BG_EXTRAS]"
else
  step "Installing llmrouter-free and corpusforge from GitHub, then BatteryGemma"
  run uv pip install \
    "llmrouter-free @ git+${LLMROUTER_REPO}@${LLMROUTER_REF}" \
    "${CF_SPEC} @ git+${CORPUSFORGE_REPO}@${CORPUSFORGE_REF}"
  run uv pip install -e ".[$BG_EXTRAS]"
fi

# ---- 4. project scaffolding --------------------------------------------------------------------------------
if [ "$SKIP_INIT" -eq 1 ]; then
  step "Skipping 'bg init' (--skip-init)"
else
  step "Scaffolding the project: configs/, .env, data directories, database"
  run "$ROOT/.venv/bin/bg" init
fi

# ---- 5. optional local model -------------------------------------------------------------------------------
step "Checking for Ollama (optional local fallback model)"
if command -v ollama >/dev/null 2>&1; then
  echo "ollama: $(command -v ollama)"
else
  echo "ollama not found - optional. Install it from https://ollama.com to give the router a local fallback"
  echo "when every free cloud provider is rate-limited (then set OLLAMA_API_KEY=any in .env)."
fi

# ---- 6. next steps -----------------------------------------------------------------------------------------
step "Done"
cat <<EOF
Activate the environment:      source .venv/bin/activate     (or prefix commands with 'uv run')
Then:
  1. Edit .env: set BG_CONTACT_EMAIL and at least one LLM provider key (e.g. OPENROUTER_API_KEY).
  2. Check the setup:          bg stats  &&  bg llm status
  3. Run the pipeline:         bg discover && bg screen && bg fetch --limit 100 && bg parse --limit 100
  4. Annotate:                 bg annotate facts --limit 200
Full walkthrough: README.md
EOF
