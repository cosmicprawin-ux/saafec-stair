"""Checkpoint helpers used by the supervised single-mutation workflow."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from core.stability_metrics import derive_val_mse_rmse
from models.stability_head import infer_head_d_model_from_state_dict


CHECKPOINT_FORMAT_VERSION = "saafec_stair_single_mutation_v1"


def capture_named_optimizer_state(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
) -> dict[str, dict[str, Any]]:
    """Return optimizer state keyed by model-parameter name."""
    param_name_by_id = {id(param): name for name, param in module.named_parameters()}
    named_state: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = param_name_by_id.get(id(param))
            if name is None:
                continue
            state = optimizer.state.get(param, {})
            named_state[name] = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
    return named_state


def build_phase_package(
    *,
    phase: int,
    epoch: int,
    head: nn.Module,
    model: nn.Module | None,
    val_mse: float,
    val_metrics: dict[str, Any],
    phase_spec: Any | None,
    train_config: dict[str, Any],
    head_optimizer_named_state: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint package written during supervised training."""
    val_mse, val_rmse = derive_val_mse_rmse(float(val_mse))
    head_state_dict = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    head_d_model = infer_head_d_model_from_state_dict(head_state_dict)
    backbone_state = {}
    if model is not None:
        backbone_state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "phase": phase,
        "epoch": epoch,
        "head_state_dict": head_state_dict,
        "head_architecture": getattr(head, "architecture_name", head.__class__.__name__),
        "head_d_model": head_d_model,
        "backbone_trainable_state": backbone_state,
        "metrics": {
            **val_metrics,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
        },
        "phase_spec": phase_spec,
        "train_config": train_config,
        "head_optimizer_named_state": head_optimizer_named_state,
        "created_at": datetime.now().isoformat(),
    }
