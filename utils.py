import json
from typing import Any, List
from typing import Optional, Tuple, Dict
from typing import Dict
import torch
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class TaskConfig:
    task_name: str
    train_file: str
    validation_file: str
    instruction: str
    options: List[str]
    text_key: str = "text"
    label_key: str = "label"
    num_epochs: float = 1.0
    output_dir: str | None = None
    max_samples: int | None = None

def normalize_text(x: str) -> str:
    return " ".join(x.strip().lower().split())

def exact_match_score(preds: List[str], refs: List[str]) -> float:
    assert len(preds) == len(refs)
    return float(sum(normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs))) / max(1, len(refs))

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_pad_token(tokenizer, model_family: str):
    if tokenizer.pad_token_id is None:
        if model_family == "causal":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # T5 usually already has pad_token
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

def infer_model_family(model_name: str, forced: Optional[str] = None) -> str:
    if forced is not None:
        return forced
    name = model_name.lower()
    if "t5" in name:
        return "seq2seq"
    if "llama" in name or "mistral" in name or "qwen" in name or "phi" in name:
        return "causal"
    raise ValueError(
        f"Could not infer model family from model_name={model_name}. "
    )

# ============================================================
# Prompt formatting
# ============================================================

def make_base_prompt(instruction: str, options: List[str], text: str) -> str:
    opts = " | ".join(options)
    return (
        f"Task Instruction: {instruction}\n"
        f"Options: {opts}\n"
        f"Text: {text}\n"
        f"Answer:"
    )

def make_seq2seq_source(instruction: str, options: List[str], text: str) -> str:
    return make_base_prompt(instruction, options, text)

def make_causal_full_text(instruction: str, options: List[str], text: str, label: str) -> str:
    return make_base_prompt(instruction, options, text) + f" {label}"

def make_causal_eval_prompt(instruction: str, options: List[str], text: str) -> str:
    return make_base_prompt(instruction, options, text)

# ============================================================
# Preprocessing
# ============================================================

def preprocess_seq2seq(
    batch: Dict[str, List[Any]],
    tokenizer,
    instruction: str,
    options: List[str],
    text_key: str,
    label_key: str,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    sources = [make_seq2seq_source(instruction, options, x) for x in batch[text_key]]
    targets = [str(x) for x in batch[label_key]]

    model_inputs = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_causal(
    batch: Dict[str, List[Any]],
    tokenizer,
    instruction: str,
    options: List[str],
    text_key: str,
    label_key: str,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    """
    Train decoder-only models on:
      prompt + " " + label

    Loss is masked so only label tokens contribute.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for text, label in zip(batch[text_key], batch[label_key]):
        prompt = make_causal_eval_prompt(instruction, options, text)
        target = " " + str(label)

        prompt_enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_source_length)
        target_enc = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_target_length)

        input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"]

        # Hard truncate total length to model max if needed
        max_total = max_source_length + max_target_length
        input_ids = input_ids[:max_total]
        attention_mask = attention_mask[:max_total]
        labels = labels[:max_total]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_task_exact_match(
    model,
    tokenizer,
    task: TaskConfig,
    model_family: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    generation_max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    """
    Generation-based evaluation on a task's validation set.
    """
    raw = load_dataset("json", data_files={"validation": task.validation_file})["validation"]

    preds: List[str] = []
    refs: List[str] = []

    model.eval()
    model.to(device)

    texts = raw[task.text_key]
    labels = raw[task.label_key]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        if model_family == "seq2seq":
            prompts = [make_seq2seq_source(task.instruction, task.options, t) for t in batch_texts]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            ).to(device)

            gen = model.generate(
                **enc,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
            )
            batch_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

        elif model_family == "causal":
            prompts = [make_causal_eval_prompt(task.instruction, task.options, t) for t in batch_texts]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            ).to(device)

            prompt_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Only decode the continuation, not the prompt
            continuation = gen[:, prompt_len:]
            batch_preds = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        else:
            raise ValueError(model_family)

        preds.extend([normalize_text(x) for x in batch_preds])
        refs.extend([normalize_text(str(x)) for x in batch_labels])

    em = exact_match_score(preds, refs)
    return {
        "task_name": task.task_name,
        "exact_match": em,
        "n_examples": len(refs),
    }

def evaluate_seen_tasks(
    model,
    tokenizer,
    tasks_seen: List[TaskConfig],
    model_family: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    generation_max_new_tokens: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], float]:
    rows = []
    for task in tasks_seen:
        res = evaluate_task_exact_match(
            model=model,
            tokenizer=tokenizer,
            task=task,
            model_family=model_family,
            batch_size=batch_size,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            generation_max_new_tokens=generation_max_new_tokens,
            device=device,
        )
        rows.append(res)
    avg_acc = float(np.mean([r["exact_match"] for r in rows])) if rows else 0.0
    return rows, avg_acc
