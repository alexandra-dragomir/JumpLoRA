from __future__ import annotations

import math
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

JumploraEllaDeltaMode = Literal["interpolated", "sparse"]


def _active_adapter_name(module: torch.nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if active is None:
        active = getattr(module, "active_adapters", None)
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    if isinstance(active, str):
        return active
    return "default"


def _resolve_jumplora_ella_delta_mode(
    model: torch.nn.Module,
    explicit: Optional[JumploraEllaDeltaMode],
) -> JumploraEllaDeltaMode:
    if explicit is not None:
        return explicit
    m = getattr(model, "_ella_jumplora_ella_delta_mode", "interpolated")
    if m not in ("interpolated", "sparse"):
        return "interpolated"
    return m  # type: ignore[return-value]


def collect_lora_deltas(
    model: torch.nn.Module,
    *,
    jumplora_ella_delta_mode: Optional[JumploraEllaDeltaMode] = None,
) -> Dict[str, torch.Tensor]:
    """Collect effective LoRA weight deltas for all LoRA layers.

    JumpLoRA: effective ΔW for the ELLA penalty follows ``jumplora_ella_delta_mode`` (or
    ``model._ella_jumplora_ella_delta_mode``): ``interpolated`` matches the forward blend;
    ``sparse`` matches the legacy branch on ``jump_interpolation_factor``.
    """
    deltas: Dict[str, torch.Tensor] = {}

    try:
        from .jump_lora import JumpLoRALinear as _JumpLoRALinear
        from .jump_lora import jumplora_linear_threshold_exp as _thr_exp
    except ImportError:
        _JumpLoRALinear = None  # type: ignore[assignment,misc]
        _thr_exp = None  # type: ignore[assignment,misc]

    mode = _resolve_jumplora_ella_delta_mode(model, jumplora_ella_delta_mode)

    for name, module in model.named_modules():
        if _JumpLoRALinear is not None and isinstance(module, _JumpLoRALinear):
            assert _thr_exp is not None
            delta_raw = module.B @ module.A
            threshold = _thr_exp(module).to(delta_raw.dtype)

            if mode == "interpolated":
                delta_jump = delta_raw * (delta_raw.abs() > threshold).to(delta_raw.dtype)
                eff = (
                    module.jump_interpolation_factor * delta_jump
                    + (1.0 - module.jump_interpolation_factor) * delta_raw
                )
                deltas[name] = eff * module.scaling
            else:
                if module.jump_interpolation_factor > 0.0:
                    delta = delta_raw * (delta_raw.abs() > threshold).to(delta_raw.dtype)
                    deltas[name] = delta * module.scaling
                else:
                    deltas[name] = delta_raw * module.scaling
            continue

        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue

        adapter = _active_adapter_name(module)
        if adapter not in lora_a or adapter not in lora_b:
            continue

        a_w = lora_a[adapter].weight
        b_w = lora_b[adapter].weight
        scaling = module.scaling.get(adapter, 1.0) if hasattr(module, "scaling") else 1.0

        delta = (b_w @ a_w) * scaling

        if getattr(module, "fan_in_fan_out", False):
            delta = delta.t()

        deltas[name] = delta

    return deltas


def reset_lora_weights(model: nn.Module) -> None:
    """Reinitialize every LoRA A/B so task t optimizes a fresh ΔW_t (ELLA paper §3.2).

    Matches PEFT defaults: Kaiming on A, zeros on B so the initial effective update is zero.
    """
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name in module.lora_A.keys():
                a = module.lora_A[adapter_name]
                b = module.lora_B[adapter_name]
                if hasattr(a, "weight") and hasattr(b, "weight"):
                    nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                    nn.init.zeros_(b.weight)
