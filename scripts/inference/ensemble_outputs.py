"""Three-seed prediction ensembling."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from inference.seed_inference import (
    CSV_DISPLAY_FIELD_NAMES,
    MUTATION_PREDICTION_FIELDS,
    mutation_prediction_fieldnames,
    write_display_rows,
)


ENSEMBLE_EXTRA_FIELDS = [
    "ensemble_n",
    "ensemble_member_predictions",
    "ensemble_member_prediction_std",
]
CSV_DISPLAY_FIELD_NAMES.update(
    {
        "ensemble_member_predictions": "ensemble_member_ΔΔG_predictions_kcal_per_mol",
        "ensemble_member_prediction_std": "ensemble_member_ΔΔG_prediction_std_kcal_per_mol",
    }
)


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("pdb", "")).strip(),
        str(row.get("chain", "")).strip(),
        str(row.get("model_position_1based", "")).strip(),
        str(row.get("wt_aa", "")).strip().upper(),
        str(row.get("mut_aa", "")).strip().upper(),
    )


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def write_rows(path: Path, rows: list[dict[str, Any]], extra_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = mutation_prediction_fieldnames(rows, extra_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        write_display_rows(handle, fieldnames, rows)


def ensemble_seed_rows(seed_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_seed = []
    for rows in seed_rows:
        keyed = {}
        for row in rows:
            key = prediction_key(row)
            if any(part == "" for part in key):
                continue
            keyed[key] = row
        by_seed.append(keyed)
    key_sets = [set(rows) for rows in by_seed]
    common = set.intersection(*key_sets) if key_sets else set()
    union = set.union(*key_sets) if key_sets else set()
    if common != union:
        missing = [len(union - keys) for keys in key_sets]
        raise RuntimeError(f"Seed prediction rows differ; missing counts by seed: {missing}")

    output = []
    for key in sorted(common):
        rows = [seed[key] for seed in by_seed]
        base = dict(rows[0])
        values = [safe_float(row.get("predicted_ddg")) for row in rows]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite seed prediction for {key}: {values}")
        mean_value = sum(values) / len(values)
        base["predicted_ddg"] = mean_value
        base["ensemble_n"] = len(values)
        base["ensemble_member_predictions"] = ";".join(format_float(value) for value in values)
        base["ensemble_member_prediction_std"] = std(values)
        output.append(base)
    return output


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
