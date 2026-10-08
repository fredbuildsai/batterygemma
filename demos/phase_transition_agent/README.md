# Demo: an agent that fixes what the models leave out

A self-contained demo (nothing in `src/batterygemma` is touched) of how much a *harness* can add to a language model,
compared with what fine-tuning alone gives.

**The problem.** Asked *"What happens to NMC811 cathodes when charged above 4.2 V?"*, the base Gemma and our fine-tuned
versions answer with generic prose ("undesirable phase changes", "lithium plating", "thermal runaway") and never name the
H1→M→H2→H3 sequence, the ~4.1-4.2 V H2→H3 transition, the c-axis collapse or the microcracking it causes - even though
all of that is in the papers we extracted facts from.

**The idea.** Don't ask the model to remember it. Let a small, readable agent (a) look the topic up in our own extracted
facts, (b) tell the model how a scientist should answer, and (c) check the answer against the evidence.

## How the agent works

```
question
  |  understand   topics.yaml: which rule fires, which search terms, which topics the answer MUST cover
  v
  |  retrieve     keyword search over the `facts` table (evidence sentences + DOI), ranked, de-duplicated, max 2 per paper
  v
  |  draft        policy.md (the behaviour) + numbered evidence + question  ->  any chat model
  v
  |  verify       tools.check_answer: every [n] exists, every cited sentence is supported by evidence n (word overlap),
  v               every must-cover topic the evidence contains appears in the answer
  |  repair       ONE revision round with the concrete problems listed (never a loop)
  v
answer + trace
```

The control flow is fixed code, not left to the model: small models are unreliable at choosing tools. The model only writes
prose. **You dictate the agent's behaviour by editing two plain files:**

| File | What you change |
|---|---|
| [`policy.md`](policy.md) | The voice and structure of the answer: cite as [n], name mechanisms precisely, the (a)-(d) layout for high-voltage questions, "never invent numbers". |
| [`topics.yaml`](topics.yaml) | Domain knowledge: which question phrases trigger which searches, material aliases, and which topics an answer must cover (`H2`, `c-axis\|c-direction`, `microcrack\|cracking`, ...). Add a rule, get a new behaviour - no Python. |

Code: [`tools.py`](tools.py) (plan, `find_facts`, `check_answer`, `ontology_lookup`), [`agent.py`](agent.py) (the loop;
`ollama:` and `cloud:` chat adapters), [`voice.py`](voice.py) (style metrics), [`run_demo.py`](run_demo.py) (comparison
report), [`test_demo.py`](test_demo.py) (12 hermetic tests).

## Run it

```bash
# from the repository root (it reads data/batterygemma.db, read-only; Ollama must be running for local models)
PYTHONPATH=demos/phase_transition_agent python demos/phase_transition_agent/run_demo.py \
    --models ollama:gemma4:e2b ollama:batterygemma-cpt ollama:batterygemma-cpt-sft \
             cloud:openrouter-nemotron-super cloud:groq-gpt-oss-120b
PYTHONPATH=demos/phase_transition_agent python demos/phase_transition_agent/run_demo.py --question "Why do Ni-rich cathodes crack?"
pytest demos/phase_transition_agent        # tests
```

`ollama:<model>` is a local model; `cloud:<deployment>` is a free-tier deployment from `configs/llm_routes.yaml` via
[llmrouter-free](https://github.com/fredbuildsai/llmrouter-free) (with a throw-away ledger, so the project database is never written).
Output goes to `results/<date>.md`: a voice table, then for each model the plain answer, the agent answer, the evidence and
the trace. Retrieval is **keyword search only** (no embeddings), by design.

## What the first run showed (2026-10-08, one question, temperature 0.2)

Full text of every answer: [`results/2026-10-08.md`](results/2026-10-08.md).

**1. Content: the harness fixes the gap; fine-tuning did not.** Without the agent, **none of the five models names the
H1/H2/H3 phases or the ~4.1 V transition** (Nemotron mentions c-axis changes and microcracks, but not the phases; the local
models mention neither). With the agent, **all five name the H1→M→H2→H3 sequence and the 4.11 V H2→H3 transition, and
connect it to oxygen oxidation and particle cracking.** Four of the five (all but CPT+SFT) also give the c-axis /
c-direction lattice shrinkage. Citations to the supporting paper ([n] → DOI) are present in every agent answer, densely
for the local models and Nemotron (3-8 each) and sparsely for gpt-oss (one). The 2B base model with the agent states the
phase sequence and voltage that the 120B cloud model leaves out when asked plainly.

**2. Voice: the harness dominates.** `register_score` (0-100, heuristics in [`voice.py`](voice.py): technical-term density,
numbers with units, citation density, prose vs bullet/header ratio, chatty phrases):

| model | plain | with agent |
|---|---|---|
| gemma4:e2b (base) | 27.6 | 83.4 |
| batterygemma-cpt | 21.5 | 84.7 |
| batterygemma-cpt-sft | 25.7 | 81.5 |
| nemotron-3-super (cloud) | 50.2 | 90.7 |
| gpt-oss-120b (cloud) | 49.0 | 57.7 |

Plain answers of every model are 75-91% bullets and headers, with chatty phrases ("here is a breakdown", "in short").
Agent answers are continuous cited prose. The gpt-oss row is the exception: it kept its bullet format despite the policy.

**3. Does fine-tuning make the voice more scientific? Not shown here.** In an earlier run of the same question the plain
scores went 26.1 → 30.0 → 36.4 (base → CPT → CPT+SFT), the trend we were hoping for; in this run they are
27.6 → 21.5 → 25.7. That the ordering flips between two runs of one prompt means **one sample cannot answer this** - the
noise is as large as the effect. Measuring it properly needs several questions × several samples per model (a `--repeats`
option is the obvious next step) and ideally a better voice measure than keyword heuristics.

**4. Where the agent struggled.** The CPT+SFT model's agent answer is the shortest (71 words, 3 citations) and it failed the "mention the c-axis" check even after the repair round (in both runs) - it follows the multi-part policy
less well than the others, consistent with the known weakness of the SFT stage. The verify step flagged it in the trace
rather than hiding it.

## Limits (be careful quoting this)

- One question, one run per model; the voice score is a transparent heuristic, not a judgement of scientific quality.
- Verification is lexical: it catches invented citations and unsupported sentences, not subtly wrong paraphrases.
- Retrieval is keyword-based; a question phrased with no trigger in `topics.yaml` fires no rule and gets no evidence
  (the agent then answers like the plain model, and the trace shows `retrieve: 0`).
- This is inference-time retrieval, a demonstration of the harness. It is not used to produce training data, consistent
  with the project's "no RAG for training" decision.
