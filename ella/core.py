from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import torch

from .lora_utils import collect_lora_deltas


@dataclass
class ELLAState:
    """Stores the accumulated past LoRA updates W_past per module."""

    past: Dict[str, torch.Tensor] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        payload = {k: v.detach().cpu() for k, v in self.past.items()}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> "ELLAState":
        payload = torch.load(path, map_location=device if device is not None else "cpu")
        if not isinstance(payload, dict):
            raise TypeError("Expected dict payload for ELLA state.")
        out: Dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            if not torch.is_tensor(value):
                raise TypeError(f"ELLA state key '{key}' is not a tensor.")
            out[key] = value
        return cls(past=out)


def compute_ella_penalty(
    deltas: Mapping[str, torch.Tensor],
    state: ELLAState,
    normalize_by_num_layers: bool = False,
) -> torch.Tensor:
    """Compute ELLA regularization term: sum || DeltaW_t ⊙ W_past ||_F^2."""
    if not deltas:
        return torch.tensor(0.0)

    total: torch.Tensor | None = None
    count = 0

    for name, delta in deltas.items():
        past = state.past.get(name)
        if past is None:
            continue
        past = past.to(device=delta.device, dtype=delta.dtype)
        term = torch.sum((delta * past) ** 2)
        total = term if total is None else total + term
        count += 1

    if total is None:
        return next(iter(deltas.values())).new_tensor(0.0)

    if normalize_by_num_layers and count > 0:
        total = total / count

    return total


def update_past_weights(
    state: ELLAState,
    deltas: Mapping[str, torch.Tensor],
) -> None:
    """Update W_past <- W_past + DeltaW_t after finishing a task."""
    for name, delta in deltas.items():
        d = delta.detach().cpu()
        if name in state.past:
            state.past[name] = state.past[name] + d
        else:
            state.past[name] = d.clone()


def compute_ella_penalty_from_model(
    model: torch.nn.Module,
    state: ELLAState,
    normalize_by_num_layers: bool = True,
) -> torch.Tensor:
    deltas = collect_lora_deltas(model)
    return compute_ella_penalty(deltas=deltas, state=state, normalize_by_num_layers=normalize_by_num_layers)


def update_past_weights_from_model(state: ELLAState, model: torch.nn.Module) -> None:
    deltas = collect_lora_deltas(model)
    update_past_weights(state=state, deltas=deltas)
