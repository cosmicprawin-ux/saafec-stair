"""Checkpoint discovery and fusion-head loading for inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.pipeline_config import resolve_output_path
from models.saprot_proteinmpnn_intrinsic_fusion import (
    LOCAL_CONTACT_CUTOFF_A,
    LOCAL_CONTACT_DISTANCE_SCALE_A,
    LOCAL_CONTACT_TOP_K,
    SaProtProteinMPNNIntrinsicFusionHead,
)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("model_config")
    if isinstance(direct, dict):
        return direct
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        nested = value.get("model_config")
        if isinstance(nested, dict):
            return nested
    return {}


def _state_and_config_from_payload(path: Path, payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(payload, dict) and "head_state_dict" in payload:
        return payload["head_state_dict"], _model_config_from_payload(payload)
    if isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload, {}
    raise TypeError(f"Unsupported single-mutation checkpoint format: {path}")


def _infer_model_config(state_dict: dict[str, torch.Tensor], model_config: dict[str, Any]) -> dict[str, Any]:
    saprot_proj = state_dict.get("saprot_proj.weight")
    if saprot_proj is not None:
        hidden_dim = int(saprot_proj.shape[0])
        d_saprot = int(saprot_proj.shape[1])
    else:
        hidden_dim = int(model_config.get("hidden_dim", 768))
        d_saprot = int(model_config.get("d_saprot", 1280))
    aa_weight = state_dict.get("aa_embedding.weight")
    aa_embed_dim = int(aa_weight.shape[1]) if aa_weight is not None else int(model_config.get("aa_embed_dim", 64))
    residual_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("residual_blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
    }
    return {
        "d_saprot": d_saprot,
        "hidden_dim": int(model_config.get("hidden_dim", hidden_dim)),
        "aa_embed_dim": aa_embed_dim,
        "dropout": float(model_config.get("dropout", 0.10)),
        "attention_heads": int(model_config.get("attention_heads", 8)),
        "residual_blocks": len(residual_indices) if residual_indices else int(model_config.get("residual_blocks", 2)),
        "local_contact_top_k": int(model_config.get("local_contact_top_k", LOCAL_CONTACT_TOP_K)),
        "local_contact_cutoff": float(model_config.get("local_contact_cutoff", LOCAL_CONTACT_CUTOFF_A)),
        "local_contact_distance_scale": float(
            model_config.get("local_contact_distance_scale", LOCAL_CONTACT_DISTANCE_SCALE_A)
        ),
    }


def load_package(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    state_dict, model_config = _state_and_config_from_payload(path, payload)
    inferred = _infer_model_config(state_dict, model_config)
    return {
        "head_state_dict": state_dict,
        "model_config": inferred,
        "d_saprot": inferred["d_saprot"],
        "checkpoint_path": str(path),
    }


def head_from_package(package: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config = package.get("model_config") or {}
    d_saprot = int(package.get("d_saprot") or model_config.get("d_saprot") or 1280)
    head = SaProtProteinMPNNIntrinsicFusionHead(
        d_saprot=d_saprot,
        d_hidden=int(model_config.get("hidden_dim", 768)),
        aa_embed_dim=int(model_config.get("aa_embed_dim", 64)),
        dropout=float(model_config.get("dropout", 0.10)),
        num_attention_heads=int(model_config.get("attention_heads", 8)),
        num_residual_blocks=int(model_config.get("residual_blocks", 2)),
        local_contact_top_k=int(model_config.get("local_contact_top_k", LOCAL_CONTACT_TOP_K)),
        local_contact_cutoff=float(model_config.get("local_contact_cutoff", LOCAL_CONTACT_CUTOFF_A)),
        local_contact_distance_scale=float(model_config.get("local_contact_distance_scale", LOCAL_CONTACT_DISTANCE_SCALE_A)),
    )
    missing, unexpected = head.load_state_dict(package["head_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Head state mismatch: missing={missing} unexpected={unexpected}")
    return head.to(device).eval()


def discover_seed_packages(checkpoint_dir: Path) -> list[Path]:
    raw_dir = Path(checkpoint_dir).expanduser()
    checkpoint_dir = raw_dir if raw_dir.exists() else resolve_output_path(raw_dir)
    patterns = [
        "seed_*/best_head.pt",
        "**/best_head.pt",
    ]
    for pattern in patterns:
        packages = sorted(checkpoint_dir.glob(pattern))
        if packages:
            return packages
    raise FileNotFoundError(f"No single-mutation seed checkpoints under {checkpoint_dir}")
