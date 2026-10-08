"""The demo agent: a fixed, inspectable pipeline around any chat model.

    understand -> retrieve -> draft -> verify -> (repair once) -> answer + trace

The control flow is deliberately NOT left to the model: small models are unreliable at choosing tools, so the policy
(policy.md), the topics (topics.yaml) and this loop decide what happens; the model only writes prose. Every step is
recorded in `Answer.trace`, which is what the README shows.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from sqlalchemy import Engine
from tools import Evidence, Verdict, check_answer, find_facts, load_topics, plan_searches

HERE = Path(__file__).parent
Chat = Callable[[list[dict[str, str]]], str]


def ollama_chat(model: str, *, host: str = "http://localhost:11434", temperature: float = 0.2,
                num_ctx: int = 8192, timeout: float = 600) -> Chat:
    """A chat function for a local Ollama model (thinking disabled, as everywhere in this project)."""
    def chat(messages: list[dict[str, str]]) -> str:
        response = httpx.post(f"{host}/api/chat", timeout=timeout, json={
            "model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        })
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    return chat


def router_chat(deployment: str, *, temperature: float = 0.2, max_tokens: int = 1500) -> Chat:
    """A chat function for ONE free-tier cloud deployment from configs/llm_routes.yaml, via llmrouter-free.

    The router gets a throw-away ledger (no engine argument), so the demo never writes to the project database.
    Failover is deliberately switched off by giving the route only this deployment: the point is to compare models.
    """
    from llmrouter_free import LLMRouter

    from batterygemma.settings import get_settings, load_config

    get_settings()  # loads .env so provider keys are in the environment
    config = load_config("llm_routes")
    chosen = [d for d in config["deployments"] if d["name"] == deployment]
    if not chosen:
        raise SystemExit(f"unknown deployment {deployment!r}; see configs/llm_routes.yaml")
    router = LLMRouter({**config, "deployments": chosen, "routes": {"demo": [deployment]}})

    def chat(messages: list[dict[str, str]]) -> str:
        return router.complete("demo", messages, temperature=temperature, max_tokens=max_tokens, use_cache=False).text.strip()
    return chat


def make_chat(spec: str) -> Chat:
    """`ollama:<model>` (local) or `cloud:<deployment>` (free tier, from configs/llm_routes.yaml)."""
    kind, _, name = spec.partition(":")
    if kind == "ollama":
        return ollama_chat(name)
    if kind == "cloud":
        return router_chat(name)
    raise SystemExit(f"model spec must look like ollama:<model> or cloud:<deployment>, got {spec!r}")


@dataclass
class Answer:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    verdict: Verdict | None = None
    repaired: bool = False
    trace: list[str] = field(default_factory=list)


def baseline(chat: Chat, question: str) -> str:
    """What the plain model says: no policy, no evidence, no checks."""
    return chat([{"role": "user", "content": question}])


def _evidence_block(evidence: list[Evidence]) -> str:
    return "\n".join(e.render() for e in evidence) or "(no evidence found)"


def ask(chat: Chat, engine: Engine, question: str, *, topics: dict | None = None, policy: str | None = None,
        k: int = 8, repair: bool = True) -> Answer:
    topics = topics or load_topics()
    policy = policy or (HERE / "policy.md").read_text(encoding="utf-8")
    trace: list[str] = []

    plan = plan_searches(question, topics)
    trace.append(f"understand: rules={plan['rules']} search_terms={len(plan['search'])} materials={plan['materials'][:3]}")

    evidence = find_facts(engine, plan["search"], plan["materials"], k=k)
    trace.append(f"retrieve: {len(evidence)} evidence sentences from {len({e.doi for e in evidence})} papers")

    messages = [
        {"role": "system", "content": policy},
        {"role": "user", "content": f"EVIDENCE\n{_evidence_block(evidence)}\n\nQUESTION\n{question}"},
    ]
    draft = chat(messages)
    trace.append(f"draft: {len(draft.split())} words")

    verdict = check_answer(draft, evidence, plan["must_mention"])
    trace.append(f"verify: ok={verdict.ok} bad_citations={verdict.bad_citations} weak={len(verdict.weakly_supported)} "
                 f"missing={verdict.missing_topics}")
    repaired = False
    if repair and not verdict.ok:
        messages += [{"role": "assistant", "content": draft},
                     {"role": "user", "content": "Revise your answer. Problems found by an automatic check:\n"
                      + verdict.feedback() + "\nKeep the same format and the [n] citations."}]
        draft = chat(messages)
        verdict = check_answer(draft, evidence, plan["must_mention"])
        repaired = True
        trace.append(f"repair: ok={verdict.ok} bad_citations={verdict.bad_citations} missing={verdict.missing_topics}")
    return Answer(draft, evidence, verdict, repaired, trace)
