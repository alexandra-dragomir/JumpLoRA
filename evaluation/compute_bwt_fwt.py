#!/usr/bin/env python3
"""BWT / FWT from continual_eval.json. See README (Evaluation). CLI: pass --continual-eval; optional --scratch-dir for FWT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent


def _norm_task(name: str) -> str:
    return str(name).strip().lower()


def load_continual_eval(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty JSON list in {path}")
    return data


def sorted_steps(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(logs, key=lambda r: int(r.get("task_index", 0)))


def task_order_from_log(logs: List[Dict[str, Any]]) -> List[str]:
    rows = sorted_steps(logs)
    return [str(r["after_training_task"]) for r in rows]


def _seen_by_task(row: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in row.get("seen_task_results") or []:
        name = _norm_task(item["task_name"])
        em = item.get("exact_match")
        if em is not None:
            out[name] = float(em)
    return out


def extract_a_tt(logs: List[Dict[str, Any]]) -> Tuple[List[str], List[Optional[float]]]:
    """Diagonal a_{t,t} after each task."""
    rows = sorted_steps(logs)
    tasks = [str(r["after_training_task"]) for r in rows]
    a_tt: List[Optional[float]] = []
    for row, task in zip(rows, tasks):
        key = _norm_task(task)
        seen = _seen_by_task(row)
        a_tt.append(seen.get(key))
    return tasks, a_tt


def extract_a_T(logs: List[Dict[str, Any]], tasks: List[str]) -> List[Optional[float]]:
    """a_{T,t} from last row."""
    rows = sorted_steps(logs)
    if not rows:
        return [None] * len(tasks)
    seen = _seen_by_task(rows[-1])
    return [seen.get(_norm_task(t)) for t in tasks]


def last_avg_seen_accuracy(logs: List[Dict[str, Any]]) -> Optional[float]:
    rows = sorted_steps(logs)
    if not rows:
        return None
    v = rows[-1].get("avg_seen_accuracy")
    return float(v) if v is not None else None


def get_metric_for_task(data: Dict[str, Any], task: str) -> Optional[float]:
    """Exact match from HF all_results.json-style keys."""
    task_l = task.lower()
    candidates = [
        f"predict_exact_match_for_{task}",
        f"predict_exact_match_for_{task_l}",
        "predict_exact_match_for_MNLI",
        "predict_exact_match_for_NLI",
        "predict_exact_match",
    ]
    for key in candidates:
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def _metric_from_scratch_continual_eval(ce_path: Path, task: str) -> Optional[float]:
    """From-scratch EM from task continual_eval.json."""
    logs = load_continual_eval(ce_path)
    rows = sorted_steps(logs)
    if not rows:
        return None
    key = _norm_task(task)
    _, a_tt = extract_a_tt(logs)
    for i, t in enumerate([str(r["after_training_task"]) for r in rows]):
        if _norm_task(t) == key and i < len(a_tt) and a_tt[i] is not None:
            return a_tt[i]
    seen = _seen_by_task(rows[-1])
    return seen.get(key)


def load_a_scratch(scratch_dir: Path, tasks: List[str]) -> List[Optional[float]]:
    """Per-task EM under scratch_dir/<task>/ from continual_eval.json or all_results.json."""
    a_scratch: List[Optional[float]] = []
    for task in tasks:
        tnorm = _norm_task(task)
        found: Optional[float] = None
        for name in (task, tnorm):
            sub = scratch_dir / name
            p_ce = sub / "continual_eval.json"
            if p_ce.is_file():
                try:
                    found = _metric_from_scratch_continual_eval(p_ce, task)
                except (ValueError, KeyError, json.JSONDecodeError):
                    found = None
                if found is not None:
                    break
            p_ar = sub / "all_results.json"
            if p_ar.is_file():
                with p_ar.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                found = get_metric_for_task(data, task)
                break
        a_scratch.append(found)
    return a_scratch


def compute_bwt(a_T: List[Optional[float]], a_tt: List[Optional[float]]) -> Optional[float]:
    t = len(a_tt)
    if t < 2:
        return None
    terms: List[float] = []
    for i in range(t - 1):
        r_t_t, r_T_t = a_tt[i], a_T[i]
        if r_t_t is not None and r_T_t is not None:
            terms.append(r_T_t - r_t_t)
    if not terms:
        return None
    return sum(terms) / (t - 1)


def compute_fwt(a_T: List[Optional[float]], a_scratch: List[Optional[float]]) -> Optional[float]:
    if not a_T or not a_scratch or len(a_T) != len(a_scratch):
        return None
    terms: List[float] = []
    for r_T_t, r_scratch_t in zip(a_T, a_scratch):
        if r_T_t is not None and r_scratch_t is not None:
            terms.append(r_T_t - r_scratch_t)
    if not terms:
        return None
    return sum(terms) / len(terms)


def fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "N/A"


def fmt_oa(x: Optional[float]) -> str:
    """OA% (metric is 0–1)."""
    return f"{100.0 * x:.4f}" if x is not None else "N/A"


def run(
    continual_eval_path: Path,
    scratch_dir: Optional[Path],
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
    logs = load_continual_eval(continual_eval_path)
    tasks, a_tt = extract_a_tt(logs)
    a_T = extract_a_T(logs, tasks)
    oa = last_avg_seen_accuracy(logs)

    bwt = compute_bwt(a_T, a_tt)

    a_scratch: Optional[List[Optional[float]]] = None
    fwt: Optional[float] = None
    if scratch_dir is not None and scratch_dir.is_dir():
        a_scratch = load_a_scratch(scratch_dir, tasks)
        fwt = compute_fwt(a_T, a_scratch)
    else:
        a_scratch = [None] * len(tasks)

    detail = {
        "continual_eval": str(continual_eval_path),
        "tasks": tasks,
        "a_tt": a_tt,
        "a_T": a_T,
        "a_scratch": a_scratch,
        "overall_avg_seen_last": oa,
        "bwt": bwt,
        "fwt": fwt,
    }
    return oa, bwt, fwt, detail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BWT/FWT from continual_eval.json")
    p.add_argument("--continual-eval", type=Path, required=True, help="continual_eval.json path")
    p.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Per-task from-scratch dirs for FWT (continual_eval.json or all_results.json per task).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full per-task arrays and paths as JSON.",
    )
    return p.parse_args()


def main() -> None:
    ns = parse_args()
    path = ns.continual_eval.resolve()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    scratch = ns.scratch_dir.resolve() if ns.scratch_dir is not None else None
    if scratch is not None and not scratch.is_dir():
        print(f"Warning: scratch-dir not a directory: {scratch} (FWT will be N/A)", file=sys.stderr)
        scratch = None

    oa, bwt, fwt, detail = run(path, scratch)

    if ns.json:
        print(json.dumps(detail, indent=2))
        return

    tasks = detail["tasks"]
    print(path)
    print("\t".join(["task", "a_{t,t}", "a_{T,t}", "a_scratch"]))
    for i, task in enumerate(tasks):
        sc_val = detail["a_scratch"][i]
        print(
            "\t".join(
                [
                    task,
                    fmt(detail["a_tt"][i]),
                    fmt(detail["a_T"][i]),
                    fmt(sc_val),
                ]
            )
        )
    print("-" * 44)
    print("\t".join(["OA% (last avg_seen)", fmt_oa(oa)]))
    print("\t".join(["BWT", fmt(bwt)]))
    print("\t".join(["FWT", fmt(fwt)]))


def _normalize_scratch(cfg_scratch: Any) -> Optional[Path]:
    if cfg_scratch is None:
        return None
    p = Path(cfg_scratch).resolve()
    return p if p.is_dir() else None


def _batch_metrics_triplet(
    ce: Path, scratch: Optional[Path]
) -> Tuple[str, str, str]:
    """(OA%, FWT, BWT) strings for one continual_eval."""
    if not ce.is_file():
        return "N/A", "N/A", "N/A"
    oa, bwt, fwt, _ = run(ce.resolve(), scratch)
    return fmt_oa(oa), fmt(fwt), fmt(bwt)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Batch: edit name/seed; no CLI args.
        _SCRATCH_LONG = ROOT_DIR / "outputs" / "tasks_from_scratch" / "long"
        name = "base_ella"
        seed = None

        scratch_long = _normalize_scratch(_SCRATCH_LONG)
        scratch_short = scratch_long

        long_header = [
            "run",
            "split",
            "o4_OA%",
            "o4_FWT",
            "o4_BWT",
            "o5_OA%",
            "o5_FWT",
            "o5_BWT",
            "o6_OA%",
            "o6_FWT",
            "o6_BWT",
        ]
        short_header = [
            "run",
            "split",
            "o1_OA%",
            "o1_FWT",
            "o1_BWT",
            "o2_OA%",
            "o2_FWT",
            "o2_BWT",
            "o3_OA%",
            "o3_FWT",
            "o3_BWT",
        ]

        long_parts: List[str] = [name, "long"]
        for o in (4, 5, 6):
            if seed is not None:
                ce = ROOT_DIR / f"outputs/{name}/long/order_{o}_{seed}/continual_eval.json"
            else:
                ce = ROOT_DIR / f"outputs/{name}/long/order_{o}/continual_eval.json"
            oa_s, fwt_s, bwt_s = _batch_metrics_triplet(ce, scratch_long)
            if oa_s == "N/A":
                print(f"  (missing continual_eval: {ce})", file=sys.stderr)
            long_parts.extend([oa_s, fwt_s, bwt_s])

        short_parts: List[str] = [name, "short"]
        for o in (1, 2, 3):
            if seed is not None:
                ce = ROOT_DIR / f"outputs/{name}/short/order_{o}_{seed}/continual_eval.json"
            else:
                ce = ROOT_DIR / f"outputs/{name}/short/order_{o}/continual_eval.json"
            oa_s, fwt_s, bwt_s = _batch_metrics_triplet(ce, scratch_short)
            if oa_s == "N/A":
                print(f"  (missing continual_eval: {ce})", file=sys.stderr)
            short_parts.extend([oa_s, fwt_s, bwt_s])

        print("\n\n")
        print("\t".join(long_header))
        print("\t".join(long_parts))
        print("-" * 44)
        print("\t".join(short_header))
        print("\t".join(short_parts))
    else:
        main()
