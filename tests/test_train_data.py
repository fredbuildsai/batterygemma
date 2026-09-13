import json

import pytest

from batterygemma.train.data import load_cpt_dataset, load_sft_dataset, render_sft_example


class StubTokenizer:
    """Mimics a HF tokenizer's apply_chat_template closely enough to test the preview helper."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is False
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_load_cpt_dataset_reads_text_field(tmp_path):
    path = tmp_path / "cpt.jsonl"
    write_jsonl(path, [{"text": "Layered oxides crack above 4.2 V."}, {"text": "SEI grows during formation."}])
    dataset = load_cpt_dataset(path)
    assert dataset.column_names == ["text"]
    assert len(dataset) == 2
    assert dataset[0]["text"] == "Layered oxides crack above 4.2 V."


def test_load_cpt_dataset_rejects_missing_or_empty_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"text": "ok"}, {"text": "   "}])
    with pytest.raises(ValueError, match="row 1"):
        load_cpt_dataset(path)


def test_load_cpt_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        load_cpt_dataset(path)


def test_render_sft_example_calls_chat_template_correctly():
    messages = [{"role": "user", "content": "Why does HF form?"}, {"role": "assistant", "content": "Hydrolysis."}]
    text = render_sft_example(messages, StubTokenizer())
    assert text == "<|user|>Why does HF form?<|assistant|>Hydrolysis."


def test_render_sft_example_rejects_examples_with_no_assistant_turn():
    with pytest.raises(ValueError, match="assistant turn"):
        render_sft_example([{"role": "user", "content": "hi"}], StubTokenizer())


def test_load_sft_dataset_keeps_messages_as_a_conversational_column(tmp_path):
    # assistant_only_loss needs the raw turn structure, not pre-rendered text - see train/sft.py docstring.
    path = tmp_path / "sft.jsonl"
    write_jsonl(path, [
        {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
        {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q2"},
                     {"role": "assistant", "content": "A2"}]},
    ])
    dataset = load_sft_dataset(path)
    assert dataset.column_names == ["messages"]
    assert len(dataset) == 2
    assert dataset[0]["messages"] == [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    assert dataset[1]["messages"][0]["content"] == "sys"


def test_load_sft_dataset_reports_which_row_is_bad(tmp_path):
    path = tmp_path / "bad_sft.jsonl"
    write_jsonl(path, [
        {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
        {"messages": [{"role": "user", "content": "no assistant turn here"}]},
    ])
    with pytest.raises(ValueError, match="row 1"):
        load_sft_dataset(path)


def test_load_sft_dataset_requires_messages_field(tmp_path):
    path = tmp_path / "no_messages.jsonl"
    write_jsonl(path, [{"text": "not an sft row"}])
    with pytest.raises(ValueError, match="'messages'"):
        load_sft_dataset(path)
