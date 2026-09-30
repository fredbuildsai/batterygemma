"""Load exported JSONL datasets (see plan: CPT `{"text": ...}`, SFT `{"messages": [...]}`) for training.

SFT rows are kept in conversational form (a `messages` column of role/content dicts) rather than pre-
rendered to a flat string: `assistant_only_loss` (this backend's response-only loss masking) requires the
raw turn structure to know where the assistant's response starts and ends - confirmed by a real training
run here, which raised "assistant_only_loss=True... only supported for conversational datasets" against a
pre-rendered `{"text": ...}` dataset. The chat template is applied by the trainer itself at train time.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from datasets import Dataset


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def load_cpt_dataset(path: Path) -> Dataset:
    rows = read_jsonl(path)
    for i, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {i} is missing a non-empty 'text' field")
    return Dataset.from_list([{"text": row["text"]} for row in rows])


def _validate_messages(messages: Any, context: str) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError(f"{context}: missing a 'messages' field")
    if not any(m.get("role") == "assistant" for m in messages):
        raise ValueError(f"{context}: SFT example has no assistant turn to train on")
    return messages


def load_sft_dataset(path: Path) -> Dataset:
    """Loads SFT rows as a conversational dataset: a `messages` column, one list of turns per row."""
    rows = read_jsonl(path)
    validated = [_validate_messages(row.get("messages"), f"{path}: row {i}") for i, row in enumerate(rows)]
    return Dataset.from_list([{"messages": m} for m in validated])


def render_sft_example(messages: list[dict[str, Any]], tokenizer: ChatTemplateTokenizer) -> str:
    """Renders one example to plain text via the tokenizer's chat template - for preview/debugging only.

    Training itself does not use this: MLXTrainer applies the chat template internally from the raw
    `messages` column, which is what `assistant_only_loss` needs (see module docstring).
    """
    _validate_messages(messages, "example")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
