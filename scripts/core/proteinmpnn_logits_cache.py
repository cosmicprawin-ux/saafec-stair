#!/usr/bin/env python3
"""Inference-time loader for cached ProteinMPNN residue logits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.pipeline_config import resolve_output_path


def resolve_proteinmpnn_cache_root(cache_dir: str | Path) -> Path:
    root = resolve_output_path(cache_dir)
    by_protein = root / "by_protein"
    return by_protein if by_protein.is_dir() else root


def _load_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ProteinMPNNLogitsCache:
    """Load per-protein ``L x 20`` ProteinMPNN logits."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = resolve_proteinmpnn_cache_root(cache_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"ProteinMPNN cache directory does not exist: {self.root}")

    def _candidate_paths(self, protein_name: str) -> list[Path]:
        protein_dir = self.root / protein_name
        return [
            protein_dir / "proteinmpnn_logits.pt",
            protein_dir / "outputs" / "proteinmpnn_logits.pt",
            protein_dir / "outputs" / "proteinmpnn" / "proteinmpnn_logits.pt",
            self.root / f"{protein_name}.pt",
        ]

    @staticmethod
    def _mask_candidates(path: Path) -> list[Path]:
        return [
            path.with_name("proteinmpnn_mask.pt"),
            path.parent / "proteinmpnn_coordinate_mask.pt",
        ]

    def load(self, protein_name: str, *, expected_length: int, wt_sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        for path in self._candidate_paths(protein_name):
            if not path.is_file():
                continue
            payload = _load_payload(path)
            mask = None
            if torch.is_tensor(payload):
                logits = payload
            elif isinstance(payload, dict):
                logits = payload.get("proteinmpnn_logits")
                if logits is None:
                    logits = payload.get("logits")
                mask = payload.get("proteinmpnn_mask")
                if mask is None:
                    mask = payload.get("coordinate_mask")
                cached_sequence = payload.get("sequence")
                if cached_sequence is not None and str(cached_sequence) != wt_sequence:
                    raise ValueError(
                        f"ProteinMPNN cache sequence mismatch for {protein_name}: "
                        f"{len(str(cached_sequence))} cached residues vs {len(wt_sequence)} dataset residues."
                    )
            else:
                logits = None
            if not torch.is_tensor(logits):
                raise TypeError(f"ProteinMPNN cache payload lacks logits tensor: {path}")
            logits = logits.detach().float().cpu()
            if logits.ndim != 2 or logits.shape[1] != 20:
                raise ValueError(f"Expected ProteinMPNN logits shape (L, 20), got {tuple(logits.shape)} at {path}")
            if logits.shape[0] != expected_length:
                raise ValueError(
                    f"ProteinMPNN length mismatch for {protein_name}: logits L={logits.shape[0]} "
                    f"but SaProt/dataset L={expected_length}."
                )
            if mask is None:
                for mask_path in self._mask_candidates(path):
                    if mask_path.is_file():
                        mask = _load_payload(mask_path)
                        break
            if mask is None:
                mask_tensor = torch.ones(expected_length, dtype=torch.bool)
            elif torch.is_tensor(mask):
                mask_tensor = mask.detach().cpu().bool().flatten()
            else:
                raise TypeError(f"ProteinMPNN mask payload is not a tensor for {protein_name}: {path}")
            if mask_tensor.shape[0] != expected_length:
                raise ValueError(
                    f"ProteinMPNN mask length mismatch for {protein_name}: mask L={mask_tensor.shape[0]} "
                    f"but logits/dataset L={expected_length}."
                )
            return logits, mask_tensor
        tried = "\n  - ".join(str(path) for path in self._candidate_paths(protein_name))
        raise FileNotFoundError(f"No ProteinMPNN logits cache found for {protein_name}. Tried:\n  - {tried}")
