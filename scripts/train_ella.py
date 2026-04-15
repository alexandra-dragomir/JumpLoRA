from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
import os
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from ella.jump_lora import inject_jumplora, merge_and_unload_jumplora, save_jumplora

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from utils import (
    preprocess_seq2seq,
    preprocess_causal,
    infer_model_family,
    ensure_pad_token,
    TaskConfig,
    save_json,
    evaluate_seen_tasks,
)

from ella.core import ELLAState, compute_ella_penalty_from_model, update_past_weights_from_model
from ella.jump_lora import JumpLoRALinear, get_model_logs

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[misc, assignment]

def load_tasks(path: str) -> List[TaskConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TaskConfig(**x) for x in raw]


def _task_with_eval_validation_basename(task: TaskConfig, basename: Optional[str]) -> TaskConfig:
    """Use ``<parent-of-task.validation_file>/<basename>`` instead of the JSON path (e.g. dev.json vs test.json)."""
    if basename is None or not str(basename).strip():
        return task
    b = str(basename).strip()
    p = Path(task.validation_file)
    new_path = p.parent / b
    if not new_path.is_file():
        alt = _PROJECT_ROOT / new_path
        if alt.is_file():
            new_path = alt
    return replace(task, validation_file=str(new_path))


class ELLATrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        ella_lambda: float,
        ella_state: ELLAState,
        wandb_step_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.wandb_step_offset = int(wandb_step_offset)
        super().__init__(*args, **kwargs)
        self.ella_lambda = ella_lambda
        self.ella_state = ella_state

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        base_loss = outputs.loss
        penalty = compute_ella_penalty_from_model(model=model, state=self.ella_state)
        ella_loss = self.ella_lambda * penalty
        loss = base_loss + ella_loss
        self._last_ce_loss = base_loss.detach()
        self._last_ella_loss = ella_loss.detach()
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Shift global_step only while logging so W&B stays monotonic across tasks."""
        if hasattr(self, "_last_ce_loss"):
            logs["train/ce_loss"] = self._last_ce_loss.item()
        if hasattr(self, "_last_ella_loss"):
            logs["train/ella_loss"] = self._last_ella_loss.item()
        o = self.wandb_step_offset
        if o:
            prev = int(self.state.global_step)
            self.state.global_step = prev + o
        try:
            super().log(logs, start_time=start_time)
        finally:
            if o:
                self.state.global_step = prev

class ELLATrainerSeq2Seq(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        ella_lambda: float,
        ella_state: ELLAState,
        wandb_step_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.wandb_step_offset = int(wandb_step_offset)
        super().__init__(*args, **kwargs)
        self.ella_lambda = ella_lambda
        self.ella_state = ella_state

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        base_loss = outputs.loss
        penalty = compute_ella_penalty_from_model(model=model, state=self.ella_state)
        ella_loss = self.ella_lambda * penalty
        loss = base_loss + ella_loss
        self._last_ce_loss = base_loss.detach()
        self._last_ella_loss = ella_loss.detach()
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if hasattr(self, "_last_ce_loss"):
            logs["train/ce_loss"] = self._last_ce_loss.item()
        if hasattr(self, "_last_ella_loss"):
            logs["train/ella_loss"] = self._last_ella_loss.item()
        o = self.wandb_step_offset
        if o:
            prev = int(self.state.global_step)
            self.state.global_step = prev + o
        try:
            super().log(logs, start_time=start_time)
        finally:
            if o:
                self.state.global_step = prev

class JumpInterpCallback(TrainerCallback):
    def __init__(
        self,
        final_interpolation_step: int,
        first_interpolation_step: int,
        model: torch.nn.Module,
        *,
        threshold_mode: str = "global",
    ):
        self.final_interpolation_step = final_interpolation_step
        self.first_interpolation_step = first_interpolation_step
        self.model = model
        self.threshold_mode = threshold_mode
        self._missing_jumplora_warned = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.final_interpolation_step = int(self.final_interpolation_step * state.max_steps)
            self.first_interpolation_step = int(self.first_interpolation_step * state.max_steps)
        interpolation_factor = (state.global_step - self.first_interpolation_step) / (
            self.final_interpolation_step - self.first_interpolation_step
        )
        interpolation_factor = min(1.0, max(0.0, interpolation_factor))
        if state.global_step == self.first_interpolation_step:
            print(f"First interpolation step: {state.global_step}")
            if self.threshold_mode == "per_block":
                by_share: defaultdict[object, List[JumpLoRALinear]] = defaultdict(list)
                for module in self.model.modules():
                    if isinstance(module, JumpLoRALinear):
                        by_share[module.threshold].append(module)
                found_any = bool(by_share)
                for share, mods in by_share.items():
                    all_flat: List[torch.Tensor] = []
                    no_updated_parameters = 0
                    no_lora_parameters = 0
                    for m in mods:
                        delta_W = (m.A.T @ m.B.T).T
                        delta_W_flat = delta_W.abs().flatten()
                        no_updated_parameters += delta_W_flat.numel()
                        no_lora_parameters += m.A.numel() + m.B.numel()
                        all_flat.append(delta_W.abs().flatten())
                    all_delta_W = torch.cat(all_flat)
                    k = max(no_updated_parameters - no_lora_parameters, 1)
                    threshold_val = torch.kthvalue(all_delta_W, k).values
                    with torch.no_grad():
                        init = threshold_val.clamp_min(1e-12).log()
                        share.log_threshold.copy_(init.to(share.log_threshold.dtype))
                if not found_any:
                    if not self._missing_jumplora_warned:
                        print(
                            "Warning: JumpInterpCallback found no JumpLoRALinear modules; "
                            "skipping threshold init/interpolation for this trainer."
                        )
                        self._missing_jumplora_warned = True
                    return
            else:
                all_delta_W: List[torch.Tensor] = []
                no_updated_parameters = 0
                no_lora_parameters = 0
                for module in self.model.modules():
                    if isinstance(module, JumpLoRALinear):
                        delta_W = (module.A.T @ module.B.T).T
                        delta_W_flattened = delta_W.abs().flatten()
                        no_updated_parameters += delta_W_flattened.numel()
                        no_lora_parameters += module.A.numel() + module.B.numel()
                        all_delta_W.append(delta_W.abs().flatten())
                if not all_delta_W:
                    if not self._missing_jumplora_warned:
                        print(
                            "Warning: JumpInterpCallback found no JumpLoRALinear modules; "
                            "skipping threshold init/interpolation for this trainer."
                        )
                        self._missing_jumplora_warned = True
                    return
                concat_delta = torch.cat(all_delta_W)
                k = no_updated_parameters - no_lora_parameters
                threshold_val = torch.kthvalue(concat_delta, k + 1).values
                gt = getattr(self.model, "global_threshold", None)
                if gt is None:
                    return
                with torch.no_grad():
                    init = threshold_val.clamp_min(1e-12).log()
                    gt.log_threshold.copy_(init.to(gt.log_threshold.dtype))
        if state.global_step <= self.final_interpolation_step:
            for module in self.model.modules():
                if isinstance(module, JumpLoRALinear):
                    module.jump_interpolation_factor = interpolation_factor


class WandbJumpLoRAThresholdSeriesCallback(TrainerCallback):
    """Log JumpLoRA thresholds aligned to monotonic train/global_step."""

    def __init__(
        self,
        model: torch.nn.Module,
        wandb_step_offset: int = 0,
        *,
        jumplora_threshold_mode: str = "global",
    ) -> None:
        self.model = model
        self.wandb_step_offset = int(wandb_step_offset)
        self.jumplora_threshold_mode = jumplora_threshold_mode

    def on_step_end(self, args, state, control, **kwargs):
        if wandb is None or wandb.run is None:
            return
        if args.logging_steps <= 0:
            return
        gs = int(state.global_step)
        if gs <= 0 or gs % args.logging_steps != 0:
            return
        model_logs = {
            f"jumplora/{k}": v.detach().float().cpu().item()
            for k, v in get_model_logs(self.model).items()
        }
        payload: Dict[str, Any] = {
            **model_logs,
            "train/global_step": gs + self.wandb_step_offset,
        }
        if self.jumplora_threshold_mode == "global":
            gt = getattr(self.model, "global_threshold", None)
            if gt is not None:
                log_t = gt.log_threshold.detach().float().cpu().item()
                payload["jumplora/train_log_threshold"] = log_t
                payload["jumplora/train_threshold"] = float(math.exp(log_t))
        wandb.log(payload)


def _wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(vars(args))
    if getattr(args, "use_jumplora", False):
        cfg["jumplora"] = {
            "bandwidth": args.jumplora_bandwidth,
            "first_interpolation_frac": args.first_interpolation_step,
            "final_interpolation_frac": args.final_interpolation_step,
            "rank": args.lora_r,
            "alpha": args.lora_alpha,
            "target_modules": list(args.target_modules) if args.target_modules else None,
            "threshold_mode": getattr(args, "jumplora_threshold_mode", "global"),
            "ella_delta_mode": getattr(args, "jumplora_ella_delta_mode", "interpolated"),
        }
    return cfg


def _jumplora_threshold_snapshot(model: torch.nn.Module, threshold_mode: str) -> Dict[str, float]:
    if threshold_mode == "global":
        gt = getattr(model, "global_threshold", None)
        if gt is None:
            return {}
        log_t = float(gt.log_threshold.detach().float().cpu().item())
        return {
            "jumplora/end_task_log_threshold": log_t,
            "jumplora/end_task_threshold": float(math.exp(log_t)),
        }
    shares = getattr(model, "jumplora_threshold_shares", None)
    if shares is None or len(shares) == 0:
        return {}
    vals = [float(t.log_threshold.detach().exp().cpu().item()) for t in shares.values()]
    return {
        "jumplora/end_task_threshold_mean": sum(vals) / len(vals),
        "jumplora/end_task_threshold_min": min(vals),
        "jumplora/end_task_threshold_max": max(vals),
    }


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_family = infer_model_family(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        task_type = TaskType.SEQ_2_SEQ_LM
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        task_type = TaskType.CAUSAL_LM

    ensure_pad_token(tokenizer, model_family)

    if args.use_jumplora:
        # JumpLoRA is injected fresh at the start of each task in the training loop.
        return model, tokenizer, None

    lora_config = LoraConfig(
        task_type=task_type,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, lora_config

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential ELLA training for LoRA adapters.")
    p.add_argument("--model", required=True, help="Base model name or path.")
    p.add_argument("--tasks", required=True, help="JSON file with a list of task specs.")
    p.add_argument("--output-root", default="outputs", help="Root output directory.")
    p.add_argument(
        "--eval-validation-basename",
        type=str,
        default=None,
        help=(
            "If set (e.g. dev.json), use that file in the same directory as each task's "
            "validation_file for the HF validation split and for continual eval — instead of "
            "the path in the task JSON (often test.json)."
        ),
    )
    p.add_argument(
        "--ella-lambda",
        type=float,
        default=3e4,
        help="ELLA lambda applied to every task when --ella-lambdas is not set.",
    )
    p.add_argument(
        "--ella-lambdas",
        nargs="+",
        type=float,
        default=None,
        metavar="LAMBDA",
        help=(
            "Optional: one ELLA lambda per task, in the same order as the task list in --tasks. "
            "Length must match the number of tasks. When set, --ella-lambda is ignored."
        ),
    )
    p.add_argument(
        "--state-path",
        default=None,
        help="Path for persisted W_past state. Default: <output-root>/ella_state.pt",
    )
    p.add_argument("--load-state", action="store_true", help="Load existing ELLA state before task 1.")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-batches before each optimizer step "
        "(effective batch ≈ batch_size * N * num_gpus).",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay (paper: 0).",
    )
    p.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="constant_with_warmup",
        help="HF scheduler name; constant_with_warmup = linear warmup then hold LR (paper-style).",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Target module names for LoRA.",
    )
    p.add_argument("--max-source-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=8)
    p.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=8,
        help="Max new tokens when generating for exact-match eval.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (Python, NumPy, PyTorch, HF Trainer dataloader).",
    )
    p.add_argument(
        "--deterministic-cudnn",
        action="store_true",
        help="Use deterministic cuDNN algorithms (slower; tighter GPU reproducibility).",
    )
    p.add_argument(
        "--use-jumplora",
        action="store_true",
        help="Use JumpLoRA (sparse LoRA via JumpReLU) instead of standard PEFT LoRA.",
    )
    p.add_argument(
        "--jumplora-bandwidth",
        type=float,
        default=0.001,
        help="JumpReLU bandwidth for the straight-through threshold gradient estimator.",
    )
    p.add_argument(
        "--first-interpolation-step",
        type=float,
        default=0.2,
        help="First step to start interpolating the JumpLoRA weights.",
    )
    p.add_argument(
        "--final-interpolation-step",
        type=float,
        default=0.8,
        help="Last step to fully interpolate the JumpLoRA weights.",
    )
    p.add_argument(
        "--jumplora-threshold-mode",
        type=str,
        default="global",
        choices=("global", "per_block"),
        help=(
            "JumpLoRA: 'global' = one learnable threshold for the whole model; "
            "'per_block' = one learnable threshold per transformer block (shared by targeted linears in that block)."
        ),
    )
    p.add_argument(
        "--jumplora-ella-delta-mode",
        type=str,
        default="interpolated",
        choices=("interpolated", "sparse"),
        help=(
            "How JumpLoRA effective ΔW is defined for ELLA regularization and W_past: "
            "'interpolated' matches the forward blend; 'sparse' uses the legacy schedule tied to jump_interpolation_factor."
        ),
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N optimizer steps in Trainer/W&B.",
    )
    p.add_argument(
        "--wandb-project",
        default="Jumplora",
        help="If set, log to Weights & Biases (single run across all CL tasks).",
    )
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run display name.")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity (user/team).")
    p.add_argument("--wandb-group", default=None, help="Optional W&B group.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.use_jumplora and args.target_modules is None:
        # LLaMA-style models use q_proj/v_proj; T5/BART-style seq2seq uses q/v on attention modules.
        mf = infer_model_family(args.model)
        if mf == "seq2seq":
            args.target_modules = ["q", "v"]
        else:
            args.target_modules = ["q_proj", "v_proj"]
        print(f"Warning: --target-modules not set; defaulting to {args.target_modules} for JumpLoRA.")

    tasks = load_tasks(args.tasks)
    if args.ella_lambdas is not None:
        if len(args.ella_lambdas) != len(tasks):
            raise ValueError(
                f"--ella-lambdas: expected {len(tasks)} values (one per task), "
                f"got {len(args.ella_lambdas)}"
            )

    model, tokenizer, lora_config = build_model_and_tokenizer(args)

    model_family = infer_model_family(args.model)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path is not None else output_root / "ella_state.pt"

    state = ELLAState()
    if args.load_state and state_path.exists():
        state = ELLAState.load(state_path)

    report_to: List[str] = []
    if args.wandb_project:
        if wandb is None:
            raise ImportError(
                "wandb is required when --wandb-project is set. Install with: pip install wandb"
            )
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=_wandb_config(args),
        )
        report_to = ["wandb"]

    if model_family == "seq2seq":
        trainer_class = ELLATrainerSeq2Seq
        train_args_cls = Seq2SeqTrainingArguments
    else:
        trainer_class = ELLATrainer
        train_args_cls = TrainingArguments

    task_train_logs: List[Dict[str, Any]] = []
    n_tasks = len(tasks)
    acc_matrix: List[List[Optional[float]]] = [[None for _ in range(n_tasks)] for _ in range(n_tasks)]
    continual_global_step_base = 0

    for idx, task in enumerate(tasks, start=1):
        ella_lam = (
            args.ella_lambdas[idx - 1] if args.ella_lambdas is not None else args.ella_lambda
        )
        print(f"\n=== Task {idx}/{len(tasks)}: {task.task_name} (ella_lambda={ella_lam}) ===")
        if args.use_jumplora:
            inject_jumplora(
                model,
                target_modules=args.target_modules,
                rank=args.lora_r,
                alpha=args.lora_alpha,
                bandwidth=args.jumplora_bandwidth,
                threshold_mode=args.jumplora_threshold_mode,
            )
            model._ella_jumplora_ella_delta_mode = args.jumplora_ella_delta_mode
        elif idx > 1:
            model = get_peft_model(model, lora_config)

        val_file = _task_with_eval_validation_basename(task, args.eval_validation_basename).validation_file
        ds = load_dataset(
            "json",
            data_files={
                "train": task.train_file,
                "validation": val_file,
            },)
        if task.max_samples is not None:
            ds["train"] = ds["train"].select(range(min(task.max_samples, len(ds["train"]))))
        task_out = Path(task.output_dir) if task.output_dir else output_root / f"task_{idx:02d}_{task.task_name}"
        task_out.mkdir(parents=True, exist_ok=True)

        if model_family == "seq2seq":
            tokenized = ds.map(
                lambda batch: preprocess_seq2seq(
                    batch=batch,
                    tokenizer=tokenizer,
                    instruction=task.instruction,
                    options=task.options,
                    text_key=task.text_key,
                    label_key=task.label_key,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                ),
                batched=True,
                remove_columns=ds["train"].column_names,
            )
            collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer,
                    model=model,
                    label_pad_token_id=-100,)
        else:
            tokenized = ds.map(
                lambda batch: preprocess_causal(
                    batch=batch,
                    tokenizer=tokenizer,
                    instruction=task.instruction,
                    options=task.options,
                    text_key=task.text_key,
                    label_key=task.label_key,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                ),
                batched=True,
                remove_columns=ds["train"].column_names,
            )
            collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

        train_args = train_args_cls(
            output_dir=str(task_out),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            # warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            logging_steps=args.logging_steps,
            # save_strategy="epoch",
            save_strategy="no",
            report_to=report_to,
            remove_unused_columns=False,
        )

        callbacks: List[TrainerCallback] = []
        if args.use_jumplora:
            callbacks.append(
                JumpInterpCallback(
                    args.final_interpolation_step,
                    args.first_interpolation_step,
                    model,
                    threshold_mode=args.jumplora_threshold_mode,
                )
            )
        if report_to and args.use_jumplora:
            callbacks.append(
                WandbJumpLoRAThresholdSeriesCallback(
                    model,
                    wandb_step_offset=continual_global_step_base,
                    jumplora_threshold_mode=args.jumplora_threshold_mode,
                )
            )

        trainer_kwargs: Dict[str, Any] = dict(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            data_collator=collator,
            ella_lambda=ella_lam,
            ella_state=state,
            wandb_step_offset=continual_global_step_base,
            callbacks=callbacks,
        )
        trainer = trainer_class(**trainer_kwargs)

        trainer.train()
        continual_global_step_base += int(trainer.state.global_step)

        jm_snap = (
            _jumplora_threshold_snapshot(model, args.jumplora_threshold_mode)
            if args.use_jumplora
            else {}
        )

        # End-of-task update: W_past <- W_past + DeltaW_t.
        update_past_weights_from_model(state=state, model=model)
        state.save(state_path)

        if args.use_jumplora:
            save_jumplora(model, task_out / "jumplora_weights.pt")
            tokenizer.save_pretrained(task_out / "adapter")
            # model = merge_and_unload_jumplora(model)
        else:
            model.save_pretrained(task_out / "adapter")
            tokenizer.save_pretrained(task_out / "adapter")
            # model = model.merge_and_unload()
        seen_tasks = tasks[:idx]
        eval_tasks = [_task_with_eval_validation_basename(t, args.eval_validation_basename) for t in seen_tasks]
        device = str(next(model.parameters()).device)
        eval_rows, avg_seen_acc = evaluate_seen_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks_seen=eval_tasks,
            model_family=model_family,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            generation_max_new_tokens=args.generation_max_new_tokens,
            device=device,
        )
        if args.use_jumplora:
            model = merge_and_unload_jumplora(model)
        else:
            model = model.merge_and_unload()

        for seen_i, row in enumerate(eval_rows):
            acc_matrix[idx - 1][seen_i] = row["exact_match"]

        step_log = {
            "after_training_task": task.task_name,
            "task_index": idx,
            "ella_lambda": ella_lam,
            "seen_task_results": eval_rows,
            "avg_seen_accuracy": avg_seen_acc,
        }
        if jm_snap:
            step_log.update(jm_snap)
        task_train_logs.append(step_log)

        if wandb is not None and wandb.run is not None:
            em_logs = {
                f"continual/exact_match/{row['task_name']}": float(row["exact_match"])
                for row in eval_rows
            }
            wandb.log(
                {
                    "continual/avg_seen_accuracy": float(avg_seen_acc),
                    "continual/task_index": idx,
                    "continual/ella_lambda": float(ella_lam),
                    "train/global_step": continual_global_step_base,
                    **em_logs,
                    **jm_snap,
                },
            )

        print(json.dumps(step_log, indent=2))
        save_json(task_train_logs, os.path.join(str(output_root), "continual_eval.json"))

    if wandb is not None and wandb.run is not None:
        wandb.finish()



if __name__ == "__main__":
    main()
