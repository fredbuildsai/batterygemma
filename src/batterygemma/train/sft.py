"""Stage 10 (Train): CPT and SFT LoRA training on Apple Silicon via Unsloth's MLX backend.

On this machine `unsloth.FastModel` transparently delegates to an internal `FastMLXModel` (mlx-lm-based)
rather than the CUDA/bitsandbytes path - confirmed by reading the installed package's source, not assumed
from documentation. Two real limits on that backend, found the same way:

- DPO is not supported: `unsloth._MLX_UNSUPPORTED_TRL_TRAINERS` explicitly lists DPOTrainer (also
  ORPOTrainer, GRPOTrainer, KTOTrainer, PPOTrainer, RewardTrainer). DPO training needs a CUDA machine
  running standard TRL's `DPOTrainer`, most likely starting from the LoRA checkpoint this module produces.
- Rendering `reasoning` into Gemma 4's thinking channel for SFT is deferred: the `gemma-4-thinking` chat
  template unconditionally strips any `<|channel>thought...<channel|>` content from *every* assistant
  message when rendering (checked by reading the template source) - it looks like a generation-time
  behavior toggle, not a mechanism meant to preserve reasoning in training targets. Revisit once stage 8
  (generate) produces real `reasoning` content to experiment against; for now SFT uses the plain `gemma-4`
  template and trains on final answers only.
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

    return FastModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length)


def apply_lora(model: Any, lora_cfg: dict[str, Any]) -> Any:
    from unsloth import FastModel

    return FastModel.get_peft_model(model, **lora_cfg)


def _build_trainer(model: Any, tokenizer: Any, dataset: Any, output_dir: Path, cfg: dict[str, Any]) -> Any:
    from unsloth import MLXTrainer, MLXTrainingConfig

    args = MLXTrainingConfig(output_dir=str(output_dir), dataset_text_field="text",
                             max_seq_length=cfg["max_seq_length"], **cfg["training"])
    return MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)


def _log_history(trainer: Any) -> list[dict[str, Any]]:
    return list(getattr(getattr(trainer, "state", None), "log_history", None) or [])


def run_cpt(dataset_path: Path, output_dir: Path, cfg: dict[str, Any]) -> TrainResult:
    """Continued pretraining: plain text, no chat template, no response masking."""
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"])
    model = apply_lora(model, cfg["lora"])
    dataset = load_cpt_dataset(dataset_path)
    trainer = _build_trainer(model, tokenizer, dataset, output_dir, cfg)
    trainer.train()
    trainer.save_model(str(output_dir))
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))


def run_sft(dataset_path: Path, output_dir: Path, cfg: dict[str, Any]) -> TrainResult:
    """SFT: chat-template-rendered messages, loss masked to assistant turns only."""
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"])
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.get("chat_template", "gemma-4"))
    model = apply_lora(model, cfg["lora"])
    dataset = load_sft_dataset(dataset_path, tokenizer)
    trainer = _build_trainer(
        model, tokenizer, dataset, output_dir,
        {**cfg, "training": {**cfg["training"], "assistant_only_loss": True,
                             "chat_template": cfg.get("chat_template", "gemma-4")}},
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))
