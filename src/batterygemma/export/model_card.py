"""The battery-specific parts of the Hugging Face model card (tags, intended use, limitations).

The card's structure and its training-data section come from `corpusforge.export.hf_release`; only the
domain text lives here.
"""

from corpusforge.export.hf_release import ModelCardInfo

BATTERY_MODEL_CARD = ModelCardInfo(
    tags=("battery-materials", "lithium-ion"),
    intended_use=(
        "Answers lithium-ion battery materials science questions and proposes research ideas grounded in "
        "the training literature. This is a proof-of-concept fine-tune, not a production or clinical-grade "
        "system - verify factual claims against primary literature before relying on them."
    ),
    limitations=(
        "Trained on a small step count as a proof of concept, not to convergence - expect uneven coverage of "
        "the underlying literature and occasional generic (non-grounded) answers.",
        "Not evaluated against a held-out gold benchmark as of this release; treat outputs as a draft "
        "requiring expert review, not an authoritative source.",
        "Domain-specific (lithium-ion/Li-metal/solid-state-Li battery materials) - not intended for "
        "general-purpose use.",
    ),
)
