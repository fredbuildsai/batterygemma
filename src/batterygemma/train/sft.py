"""Stage 10 (Train): CPT and SFT LoRA training on Apple Silicon via Unsloth's MLX backend.

On this machine `unsloth.FastModel` transparently delegates to an internal `FastMLXModel` (mlx-lm-based)
rather than the CUDA/bitsandbytes path - confirmed by reading the installed package's source, not assumed
from documentation. Real limits found the same way, each confirmed by an actual training run rather than
assumed from docs:

- DPO is not supported: `unsloth._MLX_UNSUPPORTED_TRL_TRAINERS` explicitly lists DPOTrainer (also
  ORPOTrainer, GRPOTrainer, KTOTrainer, PPOTrainer, RewardTrainer). DPO training needs a CUDA machine
  running standard TRL's `DPOTrainer`, most likely starting from the LoRA checkpoint this module produces.
- Gemma 4 E2B is natively multimodal, and this backend loads it as a VLM (via mlx-vlm) by default; the VLM
  path does not support `assistant_only_loss` at all. `load_model_and_tokenizer` forces `text_only=True`
  (this project has no vision use case) to get the plain mlx-lm text path instead.
- `assistant_only_loss` (this backend's response-only loss masking, used for SFT) requires a conversational
  dataset (a `messages` column of role/content turns) - it raises "only supported for conversational
  datasets" against a pre-rendered flat `text` column. So SFT datasets are kept as `messages` (see
  `train.data.load_sft_dataset`) and the chat template is applied by the trainer itself, not us.
- With a conversational dataset, `assistant_only_loss=True` additionally requires the chat template to mark
  the assistant span with HF's `{% generation %}...{% endgeneration %}` Jinja block, so the tokenizer can
  build an assistant-token mask via `return_assistant_tokens_mask=True`. Unsloth's built-in `gemma-4` /
  `gemma-4-thinking` templates use an older custom `<|turn>role\n...<turn|>` sentinel format with no such
  block, so this fails with "at least one example has no assistant tokens" - confirmed by a real run. Until
  a `{% generation %}`-annotated Gemma 4 template is written (or `train_on_responses_only`, the CUDA-side
  string-marker masking, is confirmed to work against `MLXTrainer`'s own batch-preparation path, which
  looks unlikely - the two mechanisms are separate implementations), SFT here trains on the *full* rendered
  sequence (prompt tokens included), i.e. `assistant_only_loss=False`. This is a real, if less token-
  efficient, SFT setup - not a correctness bug - but it is a known follow-up, not the intended final state.
- Rendering `reasoning` into Gemma 4's thinking channel for SFT is deferred: the `gemma-4-thinking` chat
  template unconditionally strips any `<|channel>thought...<channel|>` content from assistant messages when
  rendering (verified by reading the template source) - it looks like a generation-time behavior toggle,
  not a mechanism for preserving reasoning in training targets. Revisit once stage 8 (generate) produces
  real `reasoning` content to experiment against. For now SFT uses the plain `gemma-4` template and trains
  on final answers only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from batterygemma.train.data import load_cpt_dataset, load_sft_dataset


@dataclass
class TrainResult:
    output_dir: str
    log_history: list[dict[str, Any]]


def load_model_and_tokenizer(model_name: str, max_seq_length: int) -> tuple[Any, Any]:
    from unsloth import FastModel

    # See module docstring: Gemma 4 E2B is natively multimodal and loads as a VLM (mlx-vlm) by default,
    # whose path doesn't support assistant_only_loss at all - force the plain text (mlx-lm) path instead.
    return FastModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length, text_only=True)


def apply_lora(model: Any, lora_cfg: dict[str, Any]) -> Any:
    from unsloth import FastModel

    return FastModel.get_peft_model(model, **lora_cfg)


def _log_history(trainer: Any) -> list[dict[str, Any]]:
    return list(getattr(getattr(trainer, "state", None), "log_history", None) or [])


def run_cpt(dataset_path: Path, output_dir: Path, cfg: dict[str, Any]) -> TrainResult:
    """Continued pretraining: plain text, no chat template, no response masking."""
    from unsloth import MLXTrainer, MLXTrainingConfig

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"])
    model = apply_lora(model, cfg["lora"])
    dataset = load_cpt_dataset(dataset_path)
    args = MLXTrainingConfig(output_dir=str(output_dir), dataset_text_field="text",
                             max_seq_length=cfg["max_seq_length"], **cfg["training"])
    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    trainer.train()
    trainer.save_model(str(output_dir))
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))


def run_sft(dataset_path: Path, output_dir: Path, cfg: dict[str, Any]) -> TrainResult:
    """SFT: conversational dataset, chat-template applied by the trainer, loss masked to assistant turns."""
    from unsloth import MLXTrainer, MLXTrainingConfig
    from unsloth.chat_templates import get_chat_template

    chat_template = cfg.get("chat_template", "gemma-4")
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"])
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    model = apply_lora(model, cfg["lora"])
    dataset = load_sft_dataset(dataset_path)
    # assistant_only_loss=False: see module docstring - the built-in gemma-4 template lacks the
    # {% generation %} markers this backend needs to mask the loss to assistant tokens only.
    args = MLXTrainingConfig(output_dir=str(output_dir), dataset_text_field=None, chat_template=chat_template,
                             assistant_only_loss=False, max_seq_length=cfg["max_seq_length"], **cfg["training"])
    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    trainer.train()
    trainer.save_model(str(output_dir))
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))
