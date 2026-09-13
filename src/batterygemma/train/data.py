"""Load exported JSONL datasets (see plan: CPT `{"text": ...}`, SFT `{"messages": [...]}`) for training.

SFT messages are rendered to plain text via the tokenizer's chat template *before* being handed to the
trainer, rather than relying on the trainer's own chat-template auto-detection from a "messages" column -
this is more verbose but every step is directly unit-testable against a stub tokenizer, with no dependency
on the real model to check the data pipeline is correct.
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


def render_sft_example(messages: list[dict[str, Any]], tokenizer: ChatTemplateTokenizer) -> str:
    if not messages or not any(m.get("role") == "assistant" for m in messages):
        raise ValueError("SFT example has no assistant turn to train on")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_sft_dataset(path: Path, tokenizer: ChatTemplateTokenizer) -> Dataset:
    rows = read_jsonl(path)
    texts: list[str] = []
    for i, row in enumerate(rows):
        messages = row.get("messages")
        if not messages:
            raise ValueError(f"{path}: row {i} is missing a 'messages' field")
        try:
            texts.append(render_sft_example(messages, tokenizer))
        except ValueError as exc:
            raise ValueError(f"{path}: row {i}: {exc}") from exc
    return Dataset.from_list([{"text": t} for t in texts])
