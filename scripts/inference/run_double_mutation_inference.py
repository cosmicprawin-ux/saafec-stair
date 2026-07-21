#!/usr/bin/env python3
"""Run the double-mutation inference head on a provided mutation table.

This inference entry point is intentionally mutation-list driven: it predicts
only the double mutations present in the input table and never enumerates
all possible residue pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.double_mutation_dataset import (  # noqa: E402
    DoubleMutationDataset,
    double_mutation_collate_fn,
    resolve_work_path,
)
from core.pipeline_config import work_path_str  # noqa: E402
from models.double_mutation_head import DoubleMutationInteractionHead  # noqa: E402


DATASET_ROOT = "examples/quickstart/double_mutation"
DEFAULT_INPUT_TABLE = f"{DATASET_ROOT}/double_mutations.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the contact-gated double-mutation inference head on mutations "
            "listed in an input table. No all-pairs enumeration is performed."
        )
    )
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--table-sheet", dest="table_sheet", default="refined_sorted")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--single-ddg-dir", required=True)
    parser.add_argument("--proteinmpnn-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_arg)


def forward_batch(
    model: DoubleMutationInteractionHead,
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = batch["embeddings"].to(device)
    single_ddg = batch["single_ddg"].to(device)
    proteinmpnn_logits = batch["proteinmpnn_logits"].to(device)
    proteinmpnn_mask = batch["proteinmpnn_mask"].to(device)
    positions = batch["positions"].to(device)
    wt_indices = batch["wt_indices"].to(device)
    mt_indices = batch["mt_indices"].to(device)
    ca_coordinates = batch["ca_coordinates"].to(device)
    return model(
        embeddings,
        single_ddg,
        positions,
        wt_indices,
        mt_indices,
        ca_coordinates=ca_coordinates,
        proteinmpnn_logits=proteinmpnn_logits,
        proteinmpnn_mask=proteinmpnn_mask,
        return_detail_components=True,
    )


@torch.no_grad()
def predict_loader(
    model: DoubleMutationInteractionHead,
    loader: DataLoader,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        pred, baseline, correction, calibrated_base, calibration_delta, interaction, contact_prior, residual_gate = forward_batch(
            model,
            batch,
            device=device,
        )
        for idx, mutation in enumerate(batch["mutation_rows"]):
            rows.append(
                {
                    "pdb": batch["pdb"],
                    "chain": batch["chain"],
                    "identifier": batch["identifiers"][idx],
                    "position_1": mutation.positions_raw[0],
                    "position_2": mutation.positions_raw[1],
                    "wt_aa_1": mutation.wt_aa[0],
                    "wt_aa_2": mutation.wt_aa[1],
                    "mt_aa_1": mutation.mt_aa[0],
                    "mt_aa_2": mutation.mt_aa[1],
                    "additive_single_baseline": float(baseline[idx].detach().cpu()),
                    "calibrated_additive_prior": float(calibrated_base[idx].detach().cpu()),
                    "calibration_delta": float(calibration_delta[idx].detach().cpu()),
                    "contact_prior": float(contact_prior[idx].detach().cpu()),
                    "residual_gate": float(residual_gate[idx].detach().cpu()),
                    "interaction_residual": float(interaction[idx].detach().cpu()),
                    "total_correction": float(correction[idx].detach().cpu()),
                    "prediction": float(pred[idx].detach().cpu()),
                }
            )
    return rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_map = [
        ("pdb", "pdb"),
        ("chain", "chain"),
        ("identifier", "identifier"),
        ("position_1", "position_1"),
        ("position_2", "position_2"),
        ("wt_aa_1", "wt_aa_1"),
        ("wt_aa_2", "wt_aa_2"),
        ("mt_aa_1", "mt_aa_1"),
        ("mt_aa_2", "mt_aa_2"),
        ("additive_single_baseline", "additive_single_baseline_DDG_kcal_per_mol"),
        ("calibrated_additive_prior", "calibrated_additive_prior_DDG_kcal_per_mol"),
        ("calibration_delta", "calibration_delta_DDG_kcal_per_mol"),
        ("contact_prior", "contact_prior_DDG_kcal_per_mol"),
        ("residual_gate", "residual_gate"),
        ("interaction_residual", "interaction_residual_DDG_kcal_per_mol"),
        ("total_correction", "total_correction_DDG_kcal_per_mol"),
        ("prediction", "predicted_DDG_kcal_per_mol"),
    ]
    fieldnames = [display_name for _, display_name in field_map]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({display_name: row.get(source_name) for source_name, display_name in field_map})


def save_table_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def save_predictions(
    output_dir: Path,
    *,
    workbook_path: str,
    table_sheet: str,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "mutation_DDG_predictions.csv"
    save_prediction_rows(predictions_path, rows)

    payload = {
        "table_sheet": table_sheet,
        "table": work_path_str(workbook_path),
        "checkpoint_path": work_path_str(checkpoint_path),
        "mutation_DDG_predictions_path": work_path_str(predictions_path),
        "n_requested_double_mutations": len(rows),
    }
    return payload


def load_model(checkpoint_path: Path, *, device: torch.device) -> tuple[DoubleMutationInteractionHead, dict[str, Any]]:
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(package["model_config"])
    model = DoubleMutationInteractionHead(
        d_model=int(config["d_model"]),
        d_hidden=int(config.get("d_hidden", 512)),
        max_scale_delta=float(config.get("max_scale_delta", 0.35)),
        max_shift=float(config.get("max_shift", 1.5)),
        max_global_scale_delta=float(config.get("max_global_scale_delta", 0.2)),
        max_global_shift=float(config.get("max_global_shift", 0.75)),
        max_interaction=float(config.get("max_interaction", 3.0)),
        contact_cutoff=float(config.get("contact_cutoff", 10.0)),
        contact_temperature=float(config.get("contact_temperature", 2.0)),
        contact_gate_floor=float(config.get("contact_gate_floor", 0.12)),
    )
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model.to(device), package


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = resolve_work_path(args.checkpoint)
    output_dir = resolve_work_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DoubleMutationDataset(
        args.input_table,
        embeddings_dir=args.embeddings_dir,
        single_ddg_dir=args.single_ddg_dir,
        proteinmpnn_cache_dir=args.proteinmpnn_cache_dir,
        sheet_name=args.table_sheet,
    )
    if len(dataset) == 0:
        raise RuntimeError("No proteins were loaded from the specified mutation table.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=double_mutation_collate_fn,
        num_workers=args.num_workers,
    )

    model, package = load_model(checkpoint_path, device=device)
    rows = predict_loader(model, loader, device=device)
    payload = save_predictions(
        output_dir,
        workbook_path=args.input_table,
        table_sheet=args.table_sheet,
        rows=rows,
        checkpoint_path=checkpoint_path,
    )
    summary = {
        "completed_at": datetime.now().isoformat(),
        "input_table": work_path_str(args.input_table),
        "table_sheet": args.table_sheet,
        "checkpoint": work_path_str(checkpoint_path),
        "checkpoint_format_version": package.get("format_version"),
        "model_config": package.get("model_config"),
        "output_dir": work_path_str(output_dir),
        "predictions": payload,
        "note": "Predicted only the double mutations listed in the input table.",
    }
    save_json(output_dir / "inference_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
