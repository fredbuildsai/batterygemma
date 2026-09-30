# 7. Deployment View

There is exactly one deployment environment: the developer's own Apple Silicon Mac.

```mermaid
flowchart TB
    subgraph "Local machine (Apple Silicon, 16GB)"
        CLI["bg CLI<br/>(Python 3.11, uv-managed venv)"]
        DB[("data/batterygemma.db<br/>SQLite, WAL mode")]
        RAW["data/raw/&lt;source&gt;/<br/>downloaded PDFs/XML"]
        EXPORT["data/export/&lt;version&gt;/<br/>JSONL for training"]
        OUTPUTS["outputs/&lt;cpt|sft&gt;/<br/>LoRA adapters"]
        OLLAMA["ollama serve<br/>:11434 (gemma4:12b/e4b, gemma3:4b)"]
        CLI --> DB
        CLI --> RAW
        CLI --> EXPORT
        CLI --> OUTPUTS
        CLI <-->|OpenAI-compatible HTTP| OLLAMA
    end
    subgraph "Cloud (free tier)"
        NVIDIA[NVIDIA NIM]
        MISTRAL[Mistral]
    end
    CLI <-->|HTTPS| NVIDIA
    CLI <-->|HTTPS| MISTRAL
    subgraph "Public sources"
        SOURCES[OpenAlex / ChemRxiv / arXiv / Europe PMC / Unpaywall]
    end
    CLI -->|HTTPS, polite crawling| SOURCES
```

No containerization, no orchestrator, no CD step — `uv run bg <command>` is the entire runtime.

**Installing.** `scripts/setup.sh` is the one-command setup: it checks Python (3.11/3.12) and git, installs `uv` if
missing, creates `.venv`, installs `llmrouter-free` and `corpusforge` from their GitHub tags and batterygemma
editable, then runs `bg init` and `bg db init`. `--dev` clones the two packages next to this repository and installs all
three editable (for working across repositories); `--with-train` adds the Unsloth extra; `--no-parse` skips the heavy
Docling dependency. Every documented command is exercised by tests via `--dry-run` (see `tests/test_setup_script.py`).

```mermaid
flowchart LR
    GH[(GitHub tags<br/>llmrouter-free v0.1.0<br/>corpusforge v0.1.0)] -->|"scripts/setup.sh"| VENV[".venv"]
    VENV --> BGCLI["bg / corpusforge / llmrouter-free CLIs"]
    SIB["--dev: sibling checkouts<br/>../llmrouter-free ../corpusforge"] -.->|editable| VENV
```

CI: each repository has a GitHub Actions workflow (pytest + ruff on Python 3.11 and 3.12). batterygemma's CI installs the
two packages from their GitHub tags, which also verifies the documented install path continuously.
Long stages are launched as background shell scripts (`scripts/*.sh`) writing timestamped logs under
`data/logs/`, watched interactively.
