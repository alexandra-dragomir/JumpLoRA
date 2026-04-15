# JumpLoRA with ELLA

Code for **JumpLorA: Sparse Adapters for Continual Learning in Large Language Models**, using the **ELLA** objective.

**ELLA** keeps a running sum of past adapter updates `W_past` and penalizes overlap between the current task’s effective update `ΔW_t` and `W_past`. **JumpLoRA** adds low-rank adapters with a learnable threshold (global or per transformer block) and an interpolation schedule from dense toward sparse updates during training.

## Setup

**pip** (minimal stack, same pins as in `cl_env.yml`):

```bash
pip install -r requirements.txt
```

**Conda** (full environment used in experiments — PyTorch + CUDA stack, extras such as `wandb`, `deepspeed`, etc.):

```bash
conda env create -f cl_env.yml
conda activate cl_env
```

## Task format

Each run takes a JSON list of tasks (`TaskConfig` in `utils.py`): `task_name`, `train_file`, `validation_file`, `instruction`, `options`, `text_key`, `label_key`, etc. Examples: `short_order_1.json`, …

## Default training settings

Unless overridden, `train_ella.py` uses: `ella_lambda=3e4`, `learning_rate=1e-3`, `batch_size=32`, `epochs=1`, `gradient_accumulation_steps=1`, `weight_decay=0`, `lr_scheduler_type=constant_with_warmup`, `lora_r=8`, `lora_alpha=32`, `lora_dropout=0.1`, `seed=42`, `max_source_length=512`, `max_target_length=8`.

If `--target-modules` is omitted with JumpLoRA, causal LMs default to `q_proj` and `v_proj`; seq2seq defaults to `q` and `v`.

## PEFT LoRA + ELLA

```bash
python scripts/train_ella.py \
  --model google-t5/t5-large \
  --tasks short_order_1.json \
  --ella-lambda 3e4 \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --output-root outputs/run1 \
  --seed 42
```

## JumpLoRA

Use JumpLoRA instead of PEFT LoRA with `--use-jumplora`. Example with **explicit defaults** (you can omit these five lines and get the same behavior):

```bash
python scripts/train_ella.py \
  --model google-t5/t5-large \
  --tasks short_order_1.json \
  --use-jumplora \
  --ella-lambda 3e4 \
  --output-root outputs/jumplora_run1 \
  --seed 42 \
  --jumplora-threshold-mode global \
  --jumplora-ella-delta-mode interpolated \
  --jumplora-bandwidth 0.001 \
  --first-interpolation-step 0.2 \
  --final-interpolation-step 0.8
```

| Flag | Default | Other choices |
|------|---------|----------------|
| `--jumplora-threshold-mode` | `global` | `per_block` |
| `--jumplora-ella-delta-mode` | `interpolated` | `sparse` (ELLA / `W_past` effective ΔW) |
| `--jumplora-bandwidth` | `0.001` | — |
| `--first-interpolation-step` | `0.2` | fraction of total training steps |
| `--final-interpolation-step` | `0.8` | fraction of total training steps |

State: `{output-root}/ella_state.pt` (override with `--state-path`; `--load-state` to load). JumpLoRA checkpoints: `jumplora_weights.pt` per task directory.

## Evaluation (BWT / FWT)

After training, `continual_eval.json` under `--output-root` lists per-step metrics. **Backward transfer (BWT)** and optional **forward transfer (FWT)** vs from-scratch baselines:

```bash
python evaluation/compute_bwt_fwt.py --continual-eval outputs/run1/continual_eval.json
python evaluation/compute_bwt_fwt.py --continual-eval path/to/continual_eval.json --scratch-dir path/to/per_task_scratch_runs
```

`--scratch-dir` should contain one folder per task with `continual_eval.json` or `all_results.json`. Run with no arguments to use the batch stub at the bottom of that script (edit paths there).

