from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import List, Literal, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ThresholdMode = Literal["global", "per_block"]

_THRESHOLD_GROUP_REGEXES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\.encoder\.block\.(\d+)\."),
    re.compile(r"\.decoder\.block\.(\d+)\."),
    re.compile(r"\.encoder\.layer\.(\d+)\."),
    re.compile(r"\.decoder\.layer\.(\d+)\."),
    re.compile(r"\.decoder\.layers\.(\d+)\."),
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.block\.(\d+)\."),
)


def threshold_group_key_from_fqn(fqn: str) -> str:
    """Prefix shared by modules under the same transformer block (per-block thresholds)."""
    dotted = f".{fqn}" if fqn else "."
    for pat in _THRESHOLD_GROUP_REGEXES:
        m = pat.search(dotted)
        if m:
            return dotted[1 : m.end()]
    return fqn


def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).type(x.dtype)


class JumpReLUF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        result = x * (x > threshold).type(x.dtype)
        ctx.save_for_backward(x, threshold)
        ctx.stepf_bandwidth = bandwidth
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold) * grad_output
        bandwidth = ctx.stepf_bandwidth
        threshold_grad = -(threshold / bandwidth) * (
            rectangle((x - threshold) / bandwidth) * grad_output
        ).mean()
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    def __init__(self, bandwidth: float):
        super().__init__()
        self.bandwidth = bandwidth

    def __call__(self, x, threshold):
        pos = JumpReLUF.apply(x, threshold, self.bandwidth)
        neg = JumpReLUF.apply(-x, threshold, self.bandwidth)
        return pos - neg


class LearnableLogThreshold(nn.Module):
    """Scalar learnable log-threshold (one global or one per transformer block)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_threshold = nn.Parameter(torch.tensor(-4.7, dtype=torch.float32))


GlobalThreshold = LearnableLogThreshold


class JumpLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        bandwidth: float,
        threshold: LearnableLogThreshold,
    ):
        super().__init__()

        self.base = base
        self.rank = rank
        self.alpha = alpha

        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features))
        self.B = nn.Parameter(torch.randn(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.delta_bias = nn.Parameter(torch.zeros(base.out_features))
        self.jump_relu = JumpReLU(bandwidth)
        self.threshold = threshold
        self.jump_interpolation_factor = 0.0

    def forward(self, x):
        delta_W = (self.A.T @ self.B.T).T
        thr = torch.exp(self.threshold.log_threshold)
        delta_W_jump = self.jump_relu(delta_W, thr)
        delta_W_interpolated = (
            self.jump_interpolation_factor * delta_W_jump
            + (1.0 - self.jump_interpolation_factor) * delta_W
        )
        output = (x @ self.base.weight.T) + self.scaling * (x @ delta_W_interpolated.T)
        if self.base.bias is not None:
            output = output + self.base.bias
        return output


def jumplora_linear_threshold_exp(module: JumpLoRALinear) -> torch.Tensor:
    """exp(log_threshold) for sparse masking (ELLA deltas, merge)."""
    return module.threshold.log_threshold.exp()


def inject_jumplora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    bandwidth: float,
    threshold_mode: ThresholdMode = "global",
) -> nn.Module:
    """Inject JumpLoRA layers. See ``threshold_mode`` for global vs per-block thresholds."""
    if threshold_mode == "global":
        return _inject_jumplora_global(model, target_modules, rank, alpha, bandwidth)
    return _inject_jumplora_per_block(model, target_modules, rank, alpha, bandwidth)


def _inject_jumplora_global(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    bandwidth: float,
) -> nn.Module:
    target_set = set(target_modules)
    shared = LearnableLogThreshold()
    model.global_threshold = shared
    n_replaced = 0
    for parent_module in model.modules():
        for child_name, child_module in list(parent_module.named_children()):
            if child_name in target_set and isinstance(child_module, nn.Linear):
                setattr(
                    parent_module,
                    child_name,
                    JumpLoRALinear(child_module, rank, alpha, bandwidth, shared),
                )
                n_replaced += 1

    if n_replaced == 0:
        raise ValueError(
            f"JumpLoRA: no layers matched target_modules={target_modules!r}. "
            "Use names that match child nn.Linear modules (e.g. q/v for T5, q_proj/v_proj for LLaMA)."
        )

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, JumpLoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
            m.threshold.log_threshold.requires_grad = True

    return model


def _inject_jumplora_per_block(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    bandwidth: float,
) -> nn.Module:
    target_set = set(target_modules)
    candidates: List[tuple[str, nn.Linear]] = []
    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = fqn.rsplit(".", 1)[-1]
        if leaf not in target_set:
            continue
        candidates.append((fqn, mod))

    if not candidates:
        raise ValueError(
            f"JumpLoRA: no layers matched target_modules={target_modules!r}. "
            "Use names that match child nn.Linear modules (e.g. q/v for T5, q_proj/v_proj for LLaMA)."
        )

    group_keys: List[str] = []
    seen_gk: set[str] = set()
    for fqn, _ in candidates:
        gk = threshold_group_key_from_fqn(fqn)
        if gk not in seen_gk:
            seen_gk.add(gk)
            group_keys.append(gk)

    shares = nn.ModuleDict()
    for i, _gk in enumerate(group_keys):
        shares[f"t{i}"] = LearnableLogThreshold()
    gk_to_share = {gk: shares[f"t{i}"] for i, gk in enumerate(group_keys)}
    model.add_module("jumplora_threshold_shares", shares)

    for fqn, linear in candidates:
        share = gk_to_share[threshold_group_key_from_fqn(fqn)]
        parent_fqn, _, name = fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        setattr(
            parent,
            name,
            JumpLoRALinear(linear, rank, alpha, bandwidth, share),
        )

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, JumpLoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
            m.threshold.log_threshold.requires_grad = True

    return model


def merge_jumplora(model: nn.Module) -> None:
    """Merge JumpLoRA sparse deltas into base.weight in-place."""
    for module in model.modules():
        if isinstance(module, JumpLoRALinear):
            with torch.no_grad():
                delta_raw = module.B @ module.A
                threshold = jumplora_linear_threshold_exp(module).to(delta_raw.dtype)
                sparse_delta = delta_raw * (delta_raw.abs() > threshold).to(delta_raw.dtype)
                module.base.weight.add_(sparse_delta * module.scaling)


def unload_jumplora(model: nn.Module) -> nn.Module:
    """Replace JumpLoRALinear wrappers with their (now-merged) base nn.Linear modules."""
    for parent_module in model.modules():
        for child_name, child_module in list(parent_module.named_children()):
            if isinstance(child_module, JumpLoRALinear):
                setattr(parent_module, child_name, child_module.base)
    return model


def merge_and_unload_jumplora(model: nn.Module) -> nn.Module:
    merge_jumplora(model)
    unload_jumplora(model)
    return model


def save_jumplora(model: nn.Module, path: str | Path) -> None:
    """Save JumpLoRA tensors (A, B, log_threshold) per module."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, JumpLoRALinear):
            state[name] = {
                "A": module.A.detach().cpu(),
                "B": module.B.detach().cpu(),
                "log_threshold": module.threshold.log_threshold.detach().cpu(),
            }
    torch.save(state, path)


def load_jumplora(model: nn.Module, path: str | Path, device=None) -> None:
    """Load checkpoint produced by ``save_jumplora``."""
    state = torch.load(path, map_location=device or "cpu")
    modules = {name: m for name, m in model.named_modules() if isinstance(m, JumpLoRALinear)}
    for name, params in state.items():
        if name not in modules:
            logger.warning("JumpLoRA state key '%s' not found in model; skipping.", name)
            continue
        m = modules[name]
        m.A.data.copy_(params["A"].to(m.A.device))
        m.B.data.copy_(params["B"].to(m.B.device))
        m.threshold.log_threshold.data.copy_(
            params["log_threshold"].to(m.threshold.log_threshold.device)
        )


def get_model_logs(model: nn.Module):
    logs = {}
    for name, module in model.named_modules():
        if isinstance(module, JumpLoRALinear):
            delta_W = (module.A.T @ module.B.T).T
            threshold = module.threshold.log_threshold.exp()
            logs.update(
                {
                    f"{name}_mean": delta_W.abs().mean(),
                    f"{name}_max": delta_W.abs().max(),
                    f"{name}_threshold": threshold,
                    f"{name}_sparsity": (delta_W.abs() < threshold).float().mean(),
                }
            )
    return logs
