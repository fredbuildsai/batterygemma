"""Stage 10 (Train): CPT and SFT LoRA training on Apple Silicon via Unsloth's MLX backend.

On this machine `unsloth.FastModel` transparently delegates to an internal `FastMLXModel` (mlx-lm-based)
rather than the CUDA/bitsandbytes path - confirmed by reading the installed package's source, not assumed
from documentation. Real limits found the same way, each confirmed by an actual training run rather than
assumed from docs:

- DPO is not supported: `unsloth._MLX_UNSUPPORTED_TRL_TRAINERS` explicitly lists DPOTrainer (also
  ORPOTrainer, GRPOTrainer, KTOTrainer, PPOTrainer, RewardTrainer). DPO training needs a CUDA machine
  running standard TRL's `DPOTrainer`, most likely starting from the LoRA checkpoint this module produces.
- Gemma 4 E2B is natively multimodal, and this backend loads it as a VLM (via mlx-vlm) by default; the VLM
  path does not support `assistant_only_loss` at all. `text_only` (a config key, default `true` in both
  `configs/train_cpt.yaml` and `train_sft.yaml`, since this project has no vision use case yet) forces the
  plain mlx-lm text path instead when left at its default. Set it to `false` only when training against an
  `include_images` export (see `export.unsloth_jsonl`) for a genuine multimodal run.
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

import csv
import json
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batterygemma.train.data import load_cpt_dataset, load_sft_dataset
from batterygemma.train.telemetry import device_snapshot, system_snapshot


@dataclass
class TrainResult:
    output_dir: str
    log_history: list[dict[str, Any]]


class StepLogger:
    """Records one row per logged training step - start time, duration, and tokens actually used in
    that step, alongside loss/lr/memory - via `MLXTrainer.add_step_callback`, then writes them all to
    `output_dir/step_log.csv`. The trainer's callback reports *cumulative* elapsed time and token
    count since training started (see `unsloth_zoo.mlx.trainer`'s `add_step_callback` docstring); this
    class diffs consecutive calls to recover each individual step's own duration and token count,
    and anchors `start_time` to wall-clock time via `run_start` (a `time.time()` taken immediately
    before `trainer.train()`), so the CSV is directly useful for a real per-step training log rather
    than just the running averages MLXTrainer prints to the console.

    Also records this process's own memory/CPU usage via `resource.getrusage`, plus system-wide
    CPU/memory/swap/battery and GPU/unified-memory figures via `train.telemetry.system_snapshot` -
    all free (no subprocess needed for the former, no sudo for either) and exact, not sampled
    independently like an earlier `sudo powermetrics`-based approach (removed - see telemetry.py's
    module docstring for why). `ru_maxrss` is bytes on macOS (this project's only supported platform
    - see settings.py/README), NOT kilobytes as on Linux, and is a running peak for the whole
    process, not a per-step delta.
    """

    def __init__(self, run_start: float) -> None:
        self.run_start = run_start
        self.rows: list[dict[str, Any]] = []
        self._prev_elapsed = 0.0
        self._prev_tokens = 0

    def __call__(self, step: int, total_steps: int, loss: float, lr: float, tokens_sec: float,
                 peak_gb: float, elapsed: float, num_tokens: int, grad_norm: float | None = None) -> None:
        duration = elapsed - self._prev_elapsed
        step_tokens = num_tokens - self._prev_tokens
        start_time = datetime.fromtimestamp(self.run_start + self._prev_elapsed, tz=timezone.utc)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.rows.append({
            "step": step,
            "total_steps": total_steps,
            "start_time": start_time.isoformat(),
            "duration_seconds": round(duration, 3),
            "tokens_this_step": step_tokens,
            "cumulative_tokens": num_tokens,
            "tokens_per_second": round(tokens_sec, 2),
            "loss": loss,
            "learning_rate": lr,
            "peak_memory_gb": round(peak_gb, 3),  # Unsloth's own figure (mx.get_peak_memory() / 1e9, decimal GB)
            "grad_norm": grad_norm,
            "process_peak_rss_gib": round(usage.ru_maxrss / 1024**3, 3),  # binary GiB, not decimal GB
            "process_cpu_user_seconds": round(usage.ru_utime, 3),
            "process_cpu_system_seconds": round(usage.ru_stime, 3),
            **system_snapshot(),
        })
        self._prev_elapsed = elapsed
        self._prev_tokens = num_tokens

    def write_csv(self, path: Path) -> None:
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


def load_model_and_tokenizer(model_name: str, max_seq_length: int, text_only: bool = True) -> tuple[Any, Any]:
    from unsloth import FastModel

    # See module docstring: Gemma 4 E2B is natively multimodal and loads as a VLM (mlx-vlm) by default,
    # whose path doesn't support assistant_only_loss at all - text_only=True (the config default) forces
    # the plain text (mlx-lm) path instead.
    return FastModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length, text_only=text_only)


def apply_lora(model: Any, lora_cfg: dict[str, Any]) -> Any:
    from unsloth import FastModel

    return FastModel.get_peft_model(model, **lora_cfg)


def load_or_apply_lora(model: Any, lora_cfg: dict[str, Any], from_adapter: Path | None) -> Any:
    """Attach LoRA: fresh (per `lora_cfg`) by default, or continued from a prior stage's adapter.

    `from_adapter` points at a previous run's output dir (e.g. `outputs/cpt`) - its own saved
    `adapter_config.json` (rank, target modules, etc.) governs the attached layers, not `lora_cfg`,
    since continuing training must reuse the same LoRA topology the saved weights were trained into.
    This is what lets CPT -> SFT chaining actually continue the same adapter rather than starting a
    second, independent one on top of the merged base.

    Freezing before `load_adapters()` is required, not optional: `mlx_lm.tuner.utils.load_adapters` only
    converts the target modules to LoRA layers, it does not freeze the rest of the base model first (both
    mlx_lm's own CLI and Unsloth's `get_peft_model` freeze before converting - callers of the bare
    `load_adapters` utility are expected to do the same). Skipping this was confirmed live to leave ~514M
    base-model parameters trainable alongside the ~24M real LoRA parameters (measured via
    `model.trainable_parameters()` immediately after `load_adapters`, vs. the ~24.2M the original LoRA
    training actually used) - training then applies LoRA-scale learning rates to the whole unfrozen base
    model, which is exactly what produced the NaN-by-step-2 divergence seen when this was first tried.

    Freeze `model.language_model` specifically, not the whole VLM wrapper: `model.freeze()` on the full
    Gemma 4 E2B object crashes with `AttributeError: 'AudioRelativePositionEmbedding' object has no
    attribute '_no_grad'` (that submodule isn't a properly-initialized `nn.Module` - a latent bug in the
    audio tower, unrelated to this fix). Since `text_only=True` never routes a forward pass through
    `vision_tower`/`audio_tower` anyway, their parameters staying nominally "trainable" is harmless (zero
    gradient, no update) - confirmed live: freezing `language_model` alone yields exactly 24,158,208
    trainable parameters, matching the original CPT run's own count precisely.
    """
    if from_adapter is not None:
        from mlx_lm.tuner.utils import load_adapters

        getattr(model, "language_model", model).freeze()
        return load_adapters(model, str(from_adapter))
    return apply_lora(model, lora_cfg)


def _log_history(trainer: Any) -> list[dict[str, Any]]:
    return list(getattr(getattr(trainer, "state", None), "log_history", None) or [])


def write_baseline_metrics(output_dir: Path) -> dict[str, Any]:
    """Captures a snapshot of every metric `StepLogger` will later record per step - device info,
    system CPU/memory/swap/battery, GPU/unified memory - *before* anything else in a training run
    (model load, quantization, LoRA setup), so per-step figures can be compared against a genuine
    "nothing happening yet" baseline rather than just against step 1 (which already reflects the
    model load's own memory footprint). Written to `output_dir/baseline_metrics.json`. Must run
    before training starts, not concurrently with it - a baseline taken mid-load would already be
    contaminated by whatever the load itself has done to memory/CPU by that point."""
    baseline = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **device_snapshot(),
        **system_snapshot(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    return baseline


def export_gguf(model: Any, tokenizer: Any, output_dir: Path, quantization_method: str = "q4_k_m") -> Path:
    """Merge LoRA into the base weights and write a GGUF file via unsloth_zoo's MLX->llama.cpp pipeline.

    Called with the same in-memory model/tokenizer right after training, to avoid re-loading and
    re-attaching LoRA adapters from disk.
    """
    gguf_dir = output_dir / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer=tokenizer, quantization_method=quantization_method)
    return gguf_dir


def export_gguf_from_adapter(model_name: str, max_seq_length: int, adapter_dir: Path, output_dir: Path,
                              quantization_method: str = "q4_k_m", text_only: bool = True) -> Path:
    """Reload the base model, re-attach a previously saved LoRA adapter, and export GGUF.

    For converting a run that already finished (adapter saved to disk by `run_cpt`/`run_sft`)
    without re-running training. `text_only` must match what the adapter was trained with.
    """
    from mlx_lm.utils import load_adapters

    model, tokenizer = load_model_and_tokenizer(model_name, max_seq_length, text_only)
    model = load_adapters(model, str(adapter_dir))
    return export_gguf(model, tokenizer, output_dir, quantization_method=quantization_method)


def run_cpt(dataset_path: Path, output_dir: Path, cfg: dict[str, Any], gguf: bool = False,
            from_adapter: Path | None = None) -> TrainResult:
    """Continued pretraining: plain text, no chat template, no response masking.

    `from_adapter`: continue training a prior stage's adapter (e.g. a previous CPT run) instead of
    starting a fresh LoRA on the plain base model - see `load_or_apply_lora`.
    """
    from unsloth import MLXTrainer, MLXTrainingConfig

    write_baseline_metrics(output_dir)
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"], cfg.get("text_only", True))
    model = load_or_apply_lora(model, cfg["lora"], from_adapter)
    dataset = load_cpt_dataset(dataset_path)
    args = MLXTrainingConfig(output_dir=str(output_dir), dataset_text_field="text",
                             max_seq_length=cfg["max_seq_length"], **cfg["training"])
    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    step_logger = StepLogger(time.time())
    trainer.add_step_callback(step_logger)
    trainer.train()
    trainer.save_model(str(output_dir))
    step_logger.write_csv(Path(output_dir) / "step_log.csv")
    if gguf:
        export_gguf(model, tokenizer, output_dir)
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))


def run_sft(dataset_path: Path, output_dir: Path, cfg: dict[str, Any], gguf: bool = False,
            from_adapter: Path | None = None) -> TrainResult:
    """SFT: conversational dataset, chat-template applied by the trainer, loss masked to assistant turns.

    `from_adapter`: continue training a prior stage's adapter (e.g. `outputs/cpt`) instead of starting
    a fresh LoRA on the plain base model - lets CPT -> SFT be chained. Leave unset to run SFT directly
    on base, so CPT-only, SFT-only and CPT+SFT can each be compared.
    """
    from unsloth import MLXTrainer, MLXTrainingConfig
    from unsloth.chat_templates import get_chat_template

    write_baseline_metrics(output_dir)
    chat_template = cfg.get("chat_template", "gemma-4")
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], cfg["max_seq_length"], cfg.get("text_only", True))
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    model = load_or_apply_lora(model, cfg["lora"], from_adapter)
    dataset = load_sft_dataset(dataset_path)
    # assistant_only_loss=False: see module docstring - the built-in gemma-4 template lacks the
    # {% generation %} markers this backend needs to mask the loss to assistant tokens only.
    args = MLXTrainingConfig(output_dir=str(output_dir), dataset_text_field=None, chat_template=chat_template,
                             assistant_only_loss=False, max_seq_length=cfg["max_seq_length"], **cfg["training"])
    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=args)
    step_logger = StepLogger(time.time())
    trainer.add_step_callback(step_logger)
    trainer.train()
    trainer.save_model(str(output_dir))
    step_logger.write_csv(Path(output_dir) / "step_log.csv")
    if gguf:
        export_gguf(model, tokenizer, output_dir)
    return TrainResult(output_dir=str(output_dir), log_history=_log_history(trainer))
