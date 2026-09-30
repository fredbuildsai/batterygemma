# BatteryGemma - Architecture Documentation (arc42)

Structure follows [arc42](https://arc42.org/). BatteryGemma is the *domain* project of a three-repository system:
[`llmrouter-free`](https://github.com/fredbuildsai/llmrouter-free) (LLM failover router) <-
[`corpusforge`](https://github.com/fredbuildsai/corpusforge) (literature-to-corpus pipeline) <- `batterygemma`
(this repository). Start with [1 Introduction](01-introduction-and-goals.md), then the
[building-block view](05-building-block-view.md) and the [decision log](09-architecture-decisions.md).

| # | Section |
|---|---|
| 1 | [Introduction and goals](01-introduction-and-goals.md) |
| 2 | [Constraints](02-constraints.md) |
| 3 | [Context and scope](03-context-and-scope.md) |
| 4 | [Solution strategy](04-solution-strategy.md) |
| 5 | [Building block view](05-building-block-view.md) |
| 6 | [Runtime view](06-runtime-view.md) |
| 7 | [Deployment view](07-deployment-view.md) |
| 8 | [Cross-cutting concepts](08-crosscutting-concepts.md) |
| 9 | [Architecture decisions](09-architecture-decisions.md) |
| 10 | [Quality requirements](10-quality-requirements.md) |
| 11 | [Risks and technical debt](11-risks-and-technical-debt.md) |
| 12 | [Glossary](12-glossary.md) |
| A | [Appendices: CLI reference, configuration files, test suite](appendices.md) |

Deep dives: [training on Apple Silicon and GGUF export](../training.md) · [license safety end to end](../licensing.md).
The user-facing overview is the [README](../../README.md).
