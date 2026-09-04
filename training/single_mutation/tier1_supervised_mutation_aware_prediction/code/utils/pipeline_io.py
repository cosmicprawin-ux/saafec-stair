#!/usr/bin/env python3
"""
pipeline_io.py
--------------
CSV and file writing utilities for SaProt outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def tensor_to_vector(tensor: torch.Tensor) -> list[float]:
    """Collapse tensor to a fixed-length 1-D vector by averaging over token dimension."""
    t = tensor.detach().float().cpu()
    if t.ndim > 0 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 0:
        v = t.reshape(1)
    elif t.ndim == 1:
        v = t
    else:
        v = t.mean(dim=0).reshape(-1)
    return [float(x) for x in v.tolist()]


def write_per_residue_csv(tensor: torch.Tensor, csv_path: Path) -> tuple[int, int]:
    """
    Save per-token/per-residue embedding to CSV.
    Rows: token_index  |  Cols: dim_1 ... dim_D
    """
    t = tensor.detach().float().cpu()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.ndim != 2:
        raise ValueError(
            f"Expected per-residue tensor of shape [1,L,D] or [L,D], got {tuple(t.shape)}"
        )
    num_tokens, num_dims = int(t.shape[0]), int(t.shape[1])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_index"] + [f"dim_{i}" for i in range(1, num_dims + 1)])
        for idx in range(num_tokens):
            writer.writerow([idx] + [float(x) for x in t[idx].tolist()])
    return num_tokens, num_dims


def write_kind_csv(kind_name: str, kind_dir: Path) -> tuple[Path, int, int]:
    """Aggregate all per-protein tensors for one output kind into a single CSV."""
    tensor_dir = kind_dir / "tensors"
    tensor_files = sorted(tensor_dir.glob("*.pt"))
    csv_path = kind_dir / f"{kind_name}_meanpooled_vectors.csv"

    rows: list[tuple[str, list[float]]] = []
    max_dim = 0
    for tensor_file in tensor_files:
        tensor = torch.load(tensor_file, map_location="cpu")
        vector = tensor_to_vector(tensor)
        max_dim = max(max_dim, len(vector))
        rows.append((tensor_file.stem, vector))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_name"] + [f"dim_{i}" for i in range(1, max_dim + 1)])
        for pdb_name, vector in rows:
            if len(vector) < max_dim:
                vector = vector + [""] * (max_dim - len(vector))
            writer.writerow([pdb_name] + vector)

    return csv_path, len(rows), max_dim


def write_processing_stats_csv(
    output_root: Path, rows: list[dict[str, str | int | float]]
) -> Path:
    """Write per-protein processing stats (status, residues, time) to CSV."""
    csv_path = output_root / "batch_processing_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_name", "status", "num_residues", "total_time_seconds", "error_message"])
        for row in sorted(rows, key=lambda x: str(x["pdb_name"])):
            writer.writerow([
                row["pdb_name"],
                row["status"],
                row["num_residues"],
                f"{float(row['total_time_seconds']):.6f}",
                row["error_message"],
            ])
    return csv_path


def annotations_to_rows(annotations: list[Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for ann in annotations:
        start = int(ann.start)
        end = int(ann.end)
        rows.append({"label": str(ann.label), "start": start, "end": end, "length": end - start + 1})
    return rows


def write_annotations_outputs(
    annotations: list[Any], json_path: Path, csv_path: Path
) -> tuple[int, int]:
    """Write function/residue annotations to both JSON and CSV."""
    rows = annotations_to_rows(annotations)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "start", "end", "length"])
        for row in rows:
            writer.writerow([row["label"], row["start"], row["end"], row["length"]])
    return len(rows), sum(int(r["length"]) for r in rows)


def write_sasa_csv(values: list[float | None], csv_path: Path) -> None:
    """Write per-residue SASA values to CSV."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["residue_index", "sasa"])
        for idx, value in enumerate(values, start=1):
            writer.writerow([idx, "" if value is None else float(value)])


def write_vector_as_csv(vector: torch.Tensor, csv_path: Path) -> int:
    """Write a 1-D tensor as a single-row CSV."""
    v = vector.detach().float().cpu().reshape(-1)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"dim_{i}" for i in range(1, v.numel() + 1)])
        writer.writerow([float(x) for x in v.tolist()])
    return int(v.numel())
