"""Run the same question through several models, with and without the agent, and write a markdown report.

    cd demos/phase_transition_agent
    python run_demo.py --models gemma4:e2b batterygemma-cpt batterygemma-cpt-sft
    python run_demo.py --question "Why do Ni-rich cathodes crack?" --models gemma4:e2b

Reads the project database (BG_DATABASE_URL / data/batterygemma.db) read-only; needs `ollama serve` with the models.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from agent import ask, baseline, make_chat
from voice import analyse

from batterygemma.db.session import get_engine

DEFAULT_MODELS = ["ollama:gemma4:e2b", "ollama:batterygemma-cpt", "ollama:batterygemma-cpt-sft",
                  "cloud:openrouter-nemotron-super", "cloud:groq-gpt-oss-120b"]
DEFAULT_QUESTION = "What happens to NMC811 cathodes when charged above 4.2 V?"
HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="ollama:<model> (local) or cloud:<deployment> (configs/llm_routes.yaml)")
    parser.add_argument("--output", type=Path, default=HERE / "results" / f"{date.today()}.md")
    parser.add_argument("-k", type=int, default=8, help="evidence sentences per question")
    args = parser.parse_args(argv)

    engine = get_engine()
    out = [f"# Agent demo: {args.question}\n", f"Generated {date.today()}. Local models via Ollama; cloud models are free-tier "
           "deployments via llmrouter-free.\n"]
    sections: list[str] = []
    table: list[tuple[str, str, dict]] = []
    for spec in args.models:
        chat = make_chat(spec)
        print(f"== {spec}: plain", file=sys.stderr)
        plain = baseline(chat, args.question)
        print(f"== {spec}: agent", file=sys.stderr)
        answer = ask(chat, engine, args.question, k=args.k)
        table += [(spec, "plain", analyse(plain).row()), (spec, "agent", analyse(answer.text).row())]
        sections += [f"\n## {spec}\n", "### Plain model\n", plain + "\n", "### With the agent\n", answer.text + "\n",
                     "**Evidence given to the model**\n"] + [f"- {e.render()}" for e in answer.evidence]
        sections += ["\n**Agent trace**\n"] + [f"- {line}" for line in answer.trace]
    cols = ["register_score", "technical_terms_per_100w", "numbers_with_units_per_100w", "scientific_cues_per_100w",
            "citations_per_100w", "bullet_or_header_lines_pct", "chatty_phrases", "avg_sentence_words", "words"]
    out += ["\n## Voice (higher register_score = more scientific; heuristics in voice.py)\n",
            "| model | mode | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    out += [f"| {m} | {mode} | " + " | ".join(str(v[c]) for c in cols) + " |" for m, mode, v in table]
    out += sections
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
