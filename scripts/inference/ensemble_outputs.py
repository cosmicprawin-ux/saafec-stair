"""Three-seed full-matrix prediction ensembling."""
from __future__ import annotations

from typing import Any

import torch


def ensemble_seed_matrices(
    seed_matrices: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not seed_matrices:
        return {}
    key_sets = [set(matrices) for matrices in seed_matrices]
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    if common != union:
        missing = [len(union - keys) for keys in key_sets]
        raise RuntimeError(f"Seed full-matrix proteins differ; missing counts by seed: {missing}")

    output: dict[str, dict[str, Any]] = {}
    for protein_name in sorted(common):
        entries = [matrices[protein_name] for matrices in seed_matrices]
        sequences = [str(entry["sequence"]) for entry in entries]
        if len(set(sequences)) != 1:
            raise RuntimeError(f"Seed sequence mismatch for {protein_name}")
        tensors = [entry["predicted_ddg_matrix"].float() for entry in entries]
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(f"Seed full-matrix shape mismatch for {protein_name}: {sorted(shapes)}")
        output[protein_name] = {
            "protein_name": protein_name,
            "sequence": sequences[0],
            "predicted_ddg_matrix": torch.stack(tensors, dim=0).mean(dim=0),
            "ensemble_n": len(tensors),
        }
    return output
