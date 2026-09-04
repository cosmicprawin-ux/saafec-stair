#!/usr/bin/env python3
"""Train gated/calibrated double-mutation corrections on exported single-DDG priors."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.double_mutation_dataset import (  # noqa: E402
    DDG_COLUMN,
    DoubleMutationDataset,
    double_mutation_collate_fn,
    resolve_work_path,
    write_dataset_audit,
)
from models.double_mutation_head import DoubleMutationInteractionHead  # noqa: E402


DATASET_ROOT = "../../data/double_mutation"
DEFAULT_TRAIN_XLSX = f"{DATASET_ROOT}/training_set/training_set.xlsx"
DEFAULT_VAL_XLSX = f"{DATASET_ROOT}/validation_set/validation_set.xlsx"
DEFAULT_TEST_XLSX = f"{DATASET_ROOT}/test_set/test_set.xlsx"


@dataclass
class MetricBundle:
    global_pearson: float
    global_spearman: float
    rmse: float
    mae: float
    n_points: int
    n_proteins: int
    median_per_protein_pearson: float
    median_per_protein_spearman: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_pearson": self.global_pearson,
            "global_spearman": self.global_spearman,
            "rmse": self.rmse,
            "mae": self.mae,
            "n_points": self.n_points,
            "n_proteins": self.n_proteins,
            "median_per_protein_pearson": self.median_per_protein_pearson,
            "median_per_protein_spearman": self.median_per_protein_spearman,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a symmetric gated/calibrated double-mutation residual head. "
            "The additive baseline is a finalized single-mutant model exported "
            "as per-protein (L,20) matrices."
        )
    )
    parser.add_argument("--train-xlsx", default=DEFAULT_TRAIN_XLSX)
    parser.add_argument("--val-xlsx", default=DEFAULT_VAL_XLSX)
    parser.add_argument("--test-xlsx", default=DEFAULT_TEST_XLSX)
    parser.add_argument("--embeddings-dir", required=True, help="SaProt by_protein cache for double-mutation proteins.")
    parser.add_argument("--single-ddg-dir", required=True, help="Output from export_saprot_local_contact_single_ddg_for_double_mutation.py.")
    parser.add_argument(
        "--proteinmpnn-cache-dir",
        required=True,
        help="ProteinMPNN logits by_protein cache for direct interaction-residual features.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline-name",
        default="saprot_650m_pdb_proteinmpnn_intrinsic_fusion_3seed_tier2_v2_ensemble",
        help="Single-mutation baseline label to store in training metadata.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--corr-loss-weight", type=float, default=0.05)
    parser.add_argument("--correction-l2-weight", type=float, default=0.005)
    parser.add_argument("--calibration-l2-weight", type=float, default=0.001)
    parser.add_argument("--interaction-l2-weight", type=float, default=0.003)
    parser.add_argument(
        "--far-interaction-l2-weight",
        type=float,
        default=0.012,
        help="Extra shrinkage for interaction residuals on low-contact-prior residue pairs.",
    )
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-test-eval", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_warmup_schedule(optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _corr(values_a: list[float], values_b: list[float], *, method: str) -> float:
    if len(values_a) < 3:
        return float("nan")
    if len(set(values_a)) < 2 or len(set(values_b)) < 2:
        return float("nan")
    if method == "pearson":
        value, _ = stats.pearsonr(values_a, values_b)
    elif method == "spearman":
        value, _ = stats.spearmanr(values_a, values_b)
    else:
        raise ValueError(method)
    return float(value)


def _median_ignore_nan(values: list[float]) -> float:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def compute_metrics(rows: list[dict[str, Any]], *, prediction_key: str) -> MetricBundle:
    pred = [float(row[prediction_key]) for row in rows]
    true = [float(row["target_ddg"]) for row in rows]
    errors = [p - t for p, t in zip(pred, true)]
    per_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_protein.setdefault(str(row["protein_name"]), []).append(row)
    pp_pearson = [
        _corr([float(row[prediction_key]) for row in protein_rows], [float(row["target_ddg"]) for row in protein_rows], method="pearson")
        for protein_rows in per_protein.values()
    ]
    pp_spearman = [
        _corr([float(row[prediction_key]) for row in protein_rows], [float(row["target_ddg"]) for row in protein_rows], method="spearman")
        for protein_rows in per_protein.values()
    ]
    return MetricBundle(
        global_pearson=_corr(pred, true, method="pearson"),
        global_spearman=_corr(pred, true, method="spearman"),
        rmse=float(math.sqrt(sum(err * err for err in errors) / max(len(errors), 1))),
        mae=float(sum(abs(err) for err in errors) / max(len(errors), 1)),
        n_points=len(rows),
        n_proteins=len(per_protein),
        median_per_protein_pearson=_median_ignore_nan(pp_pearson),
        median_per_protein_spearman=_median_ignore_nan(pp_spearman),
    )


def centered_pearson_loss(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if pred.numel() < 3:
        return pred.new_tensor(0.0)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    target_var = torch.sum(target_centered.square())
    if float(target_var.detach().cpu()) <= eps:
        return pred.new_tensor(0.0)
    pred_var = torch.sum(pred_centered.square()).clamp_min(eps)
    denom = torch.sqrt(pred_var * target_var.clamp_min(eps))
    corr = torch.sum(pred_centered * target_centered) / denom
    return 1.0 - corr.clamp(min=-1.0, max=1.0)


def per_protein_metric_rows(rows: list[dict[str, Any]], *, prediction_key: str) -> list[dict[str, Any]]:
    per_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_protein.setdefault(str(row["protein_name"]), []).append(row)

    output: list[dict[str, Any]] = []
    for protein_name, protein_rows in sorted(per_protein.items()):
        pred = [float(row[prediction_key]) for row in protein_rows]
        true = [float(row["target_ddg"]) for row in protein_rows]
        errors = [p - t for p, t in zip(pred, true)]
        output.append(
            {
                "protein_name": protein_name,
                "pearson": _corr(pred, true, method="pearson"),
                "spearman": _corr(pred, true, method="spearman"),
                "rmse": float(math.sqrt(sum(err * err for err in errors) / max(len(errors), 1))),
                "mae": float(sum(abs(err) for err in errors) / max(len(errors), 1)),
                "n_mutations": len(protein_rows),
            }
        )
    return output


def forward_batch(
    model: DoubleMutationInteractionHead,
    batch: dict[str, Any],
    *,
    device: torch.device,
    return_components: bool = False,
    return_detail_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return_components=return_components,
        return_detail_components=return_detail_components,
    )


def train_epoch(
    model: DoubleMutationInteractionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    *,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
    corr_loss_weight: float,
    correction_l2_weight: float,
    calibration_l2_weight: float,
    interaction_l2_weight: float,
    far_interaction_l2_weight: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader, start=1):
        targets = batch["targets"].to(device)
        pred, _baseline, correction, _calibrated_base, calibration_delta, interaction, contact_prior, _residual_gate = forward_batch(
            model,
            batch,
            device=device,
            return_detail_components=True,
        )
        mse_loss = nn.functional.mse_loss(pred, targets)
        corr_loss = centered_pearson_loss(pred, targets)
        correction_reg = torch.mean(correction.square())
        calibration_reg = torch.mean(calibration_delta.square())
        interaction_reg = torch.mean(interaction.square())
        far_interaction_reg = torch.mean(interaction.square() * (1.0 - contact_prior.detach()))
        loss = (
            mse_loss
            + float(corr_loss_weight) * corr_loss
            + float(correction_l2_weight) * correction_reg
            + float(calibration_l2_weight) * calibration_reg
            + float(interaction_l2_weight) * interaction_reg
            + float(far_interaction_l2_weight) * far_interaction_reg
        )
        (loss / grad_accum).backward()
        total_loss += float(loss.item())
        steps += 1
        if batch_idx % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    if steps % grad_accum:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loader(
    model: DoubleMutationInteractionHead,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, MetricBundle, MetricBundle, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    rows: list[dict[str, Any]] = []
    for batch in loader:
        targets = batch["targets"].to(device)
        pred, baseline, correction, calibrated_base, calibration_delta, interaction, contact_prior, residual_gate = forward_batch(
            model,
            batch,
            device=device,
            return_detail_components=True,
        )
        loss = nn.functional.mse_loss(pred, targets)
        total_loss += float(loss.item())
        n_batches += 1
        for idx, mutation in enumerate(batch["mutation_rows"]):
            rows.append(
                {
                    "protein_name": batch["protein_name"],
                    "pdb": batch["pdb"],
                    "chain": batch["chain"],
                    "identifier": batch["identifiers"][idx],
                    "position_1": mutation.positions_raw[0],
                    "position_2": mutation.positions_raw[1],
                    "wt_aa_1": mutation.wt_aa[0],
                    "wt_aa_2": mutation.wt_aa[1],
                    "mt_aa_1": mutation.mt_aa[0],
                    "mt_aa_2": mutation.mt_aa[1],
                    "target_ddg": float(targets[idx].detach().cpu()),
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
    return (
        total_loss / max(n_batches, 1),
        compute_metrics(rows, prediction_key="prediction"),
        compute_metrics(rows, prediction_key="additive_single_baseline"),
        rows,
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protein_name",
        "pdb",
        "chain",
        "identifier",
        "position_1",
        "position_2",
        "wt_aa_1",
        "wt_aa_2",
        "mt_aa_1",
        "mt_aa_2",
        "target_ddg",
        "additive_single_baseline",
        "calibrated_additive_prior",
        "calibration_delta",
        "contact_prior",
        "residual_gate",
        "interaction_residual",
        "total_correction",
        "prediction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def save_history_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_split_evaluation(
    output_dir: Path,
    *,
    split_name: str,
    split_role: str,
    workbook_path: str,
    loss: float,
    metrics: MetricBundle,
    baseline_metrics: MetricBundle,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    epoch: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "mutation_predictions.csv"
    save_prediction_rows(predictions_path, rows)

    per_protein_path = output_dir / "per_protein_metrics.csv"
    save_table_rows(
        per_protein_path,
        per_protein_metric_rows(rows, prediction_key="prediction"),
        fieldnames=["protein_name", "pearson", "spearman", "rmse", "mae", "n_mutations"],
    )

    payload = {
        "phase": 1,
        "phase_name": "double_mutation_contact_gated_calibrated_prior_residual_head",
        "epoch": epoch,
        "split": split_name,
        "split_role": split_role,
        "dataset": split_name,
        "xlsx_sheet": "refined_sorted",
        "table": workbook_path,
        "checkpoint_path": str(checkpoint_path),
        "mutation_predictions_path": str(predictions_path),
        "per_protein_metrics_path": str(per_protein_path),
        "loss": loss,
        "double_head": metrics.to_dict(),
        "additive_single_baseline": baseline_metrics.to_dict(),
        "global_pearson": metrics.global_pearson,
        "global_spearman": metrics.global_spearman,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "median_per_protein_pearson": metrics.median_per_protein_pearson,
        "median_per_protein_spearman": metrics.median_per_protein_spearman,
        "n_proteins": metrics.n_proteins,
        "n_mutations": metrics.n_points,
        "baseline_global_pearson": baseline_metrics.global_pearson,
        "baseline_global_spearman": baseline_metrics.global_spearman,
        "baseline_rmse": baseline_metrics.rmse,
        "baseline_mae": baseline_metrics.mae,
    }
    save_json(output_dir / "metrics.json", payload)
    save_table_rows(
        output_dir / "metrics_summary.csv",
        [payload],
        fieldnames=[
            "phase",
            "phase_name",
            "epoch",
            "split",
            "split_role",
            "xlsx_sheet",
            "table",
            "checkpoint_path",
            "mutation_predictions_path",
            "loss",
            "global_pearson",
            "global_spearman",
            "rmse",
            "mae",
            "median_per_protein_pearson",
            "median_per_protein_spearman",
            "n_proteins",
            "n_mutations",
            "baseline_global_pearson",
            "baseline_global_spearman",
            "baseline_rmse",
            "baseline_mae",
        ],
    )
    return payload


def package_state(
    model: DoubleMutationInteractionHead,
    *,
    epoch: int,
    val_loss: float,
    val_metrics: MetricBundle,
    baseline_metrics: MetricBundle,
    train_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": "double_mutation_contact_gated_calibrated_prior_saprot_proteinmpnn_v1",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "architecture": model.architecture_name,
            "d_model": model.d_model,
            "d_hidden": model.d_hidden,
            "max_scale_delta": model.max_scale_delta,
            "max_shift": model.max_shift,
            "max_global_scale_delta": model.max_global_scale_delta,
            "max_global_shift": model.max_global_shift,
            "max_interaction": model.max_interaction,
            "contact_cutoff": model.contact_cutoff,
            "contact_temperature": model.contact_temperature,
            "contact_gate_floor": model.contact_gate_floor,
        },
        "metrics": {
            "val_loss": val_loss,
            "double_head": val_metrics.to_dict(),
            "additive_single_baseline": baseline_metrics.to_dict(),
        },
        "train_config": train_config,
    }


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    device = choose_device(args.device)
    output_dir = resolve_work_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    train_ds = DoubleMutationDataset(
        args.train_xlsx,
        embeddings_dir=args.embeddings_dir,
        single_ddg_dir=args.single_ddg_dir,
        proteinmpnn_cache_dir=args.proteinmpnn_cache_dir,
        split="train",
    )
    val_ds = DoubleMutationDataset(
        args.val_xlsx,
        embeddings_dir=args.embeddings_dir,
        single_ddg_dir=args.single_ddg_dir,
        proteinmpnn_cache_dir=args.proteinmpnn_cache_dir,
        split="validation",
    )
    test_ds = None if args.skip_test_eval else DoubleMutationDataset(
        args.test_xlsx,
        embeddings_dir=args.embeddings_dir,
        single_ddg_dir=args.single_ddg_dir,
        proteinmpnn_cache_dir=args.proteinmpnn_cache_dir,
        split="testing",
    )
    if len(train_ds) == 0:
        raise RuntimeError("Training double-mutation dataset is empty.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation double-mutation dataset is empty.")

    write_dataset_audit(
        output_dir / "dataset_audit.json",
        {"training": train_ds, "validation": val_ds, **({} if test_ds is None else {"testing": test_ds})},
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=double_mutation_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=double_mutation_collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = None if test_ds is None else DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=double_mutation_collate_fn,
        num_workers=args.num_workers,
    )

    first_sample = train_ds[0]
    d_model = int(first_sample.embeddings.shape[1])
    model = DoubleMutationInteractionHead(
        d_model=d_model,
        d_hidden=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = cosine_warmup_schedule(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_frac),
        total_steps=total_steps,
    )

    train_config = {
        "workflow": "double_saprot_650m_pdb_proteinmpnn_intrinsic_tier2_contact_gated_calibrated_prior_mpnn_residual_3seed_v2",
        "baseline": args.baseline_name,
        "architecture": model.architecture_name,
        "train_xlsx": args.train_xlsx,
        "val_xlsx": args.val_xlsx,
        "test_xlsx": None if args.skip_test_eval else args.test_xlsx,
        "embeddings_dir": args.embeddings_dir,
        "single_ddg_dir": args.single_ddg_dir,
        "proteinmpnn_cache_dir": args.proteinmpnn_cache_dir,
        "output_dir": str(output_dir),
        "d_model": d_model,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "corr_loss_weight": args.corr_loss_weight,
        "correction_l2_weight": args.correction_l2_weight,
        "calibration_l2_weight": args.calibration_l2_weight,
        "interaction_l2_weight": args.interaction_l2_weight,
        "far_interaction_l2_weight": args.far_interaction_l2_weight,
        "grad_accum": args.grad_accum,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_selection_metric": "validation_global_pearson",
        "train_proteins": len(train_ds),
        "val_proteins": len(val_ds),
        "test_proteins": None if test_ds is None else len(test_ds),
        "started_at": datetime.now().isoformat(),
    }
    save_json(output_dir / "train_config.json", train_config)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            grad_accum=args.grad_accum,
            grad_clip=args.grad_clip,
            corr_loss_weight=args.corr_loss_weight,
            correction_l2_weight=args.correction_l2_weight,
            calibration_l2_weight=args.calibration_l2_weight,
            interaction_l2_weight=args.interaction_l2_weight,
            far_interaction_l2_weight=args.far_interaction_l2_weight,
        )
        val_loss, val_metrics, baseline_metrics, val_rows = evaluate_loader(
            model,
            val_loader,
            device=device,
        )
        score = val_metrics.global_pearson if val_metrics.global_pearson == val_metrics.global_pearson else float("-inf")
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            package = package_state(
                model,
                epoch=epoch,
                val_loss=val_loss,
                val_metrics=val_metrics,
                baseline_metrics=baseline_metrics,
                train_config=train_config,
            )
            best_checkpoint_path = checkpoints_dir / "best_double_mutation_package.pt"
            torch.save(package, best_checkpoint_path)
            save_json(output_dir / "best_val_metrics.json", package["metrics"])
            save_prediction_rows(output_dir / "validation_predictions_best.csv", val_rows)
            val_payload = save_split_evaluation(
                output_dir / "validation" / "validation_set" / "phase1_best",
                split_name="validation_set",
                split_role="validation",
                workbook_path=args.val_xlsx,
                loss=val_loss,
                metrics=val_metrics,
                baseline_metrics=baseline_metrics,
                rows=val_rows,
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
            )
            save_table_rows(
                output_dir / "best_metrics_summary.csv",
                [val_payload],
                fieldnames=[
                    "phase",
                    "phase_name",
                    "epoch",
                    "split",
                    "split_role",
                    "loss",
                    "global_pearson",
                    "global_spearman",
                    "rmse",
                    "mae",
                    "median_per_protein_pearson",
                    "median_per_protein_spearman",
                    "n_proteins",
                    "n_mutations",
                    "baseline_global_pearson",
                    "baseline_global_spearman",
                    "baseline_rmse",
                    "baseline_mae",
                ],
            )
        else:
            patience_counter += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_global_pearson": val_metrics.global_pearson,
            "val_global_spearman": val_metrics.global_spearman,
            "val_rmse": val_metrics.rmse,
            "val_mae": val_metrics.mae,
            "baseline_val_global_pearson": baseline_metrics.global_pearson,
            "baseline_val_global_spearman": baseline_metrics.global_spearman,
            "baseline_val_rmse": baseline_metrics.rmse,
            "baseline_val_mae": baseline_metrics.mae,
            "best_score": best_score,
            "is_best": improved,
            "elapsed_sec": perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_pearson={val_metrics.global_pearson:.4f} "
            f"baseline_val_pearson={baseline_metrics.global_pearson:.4f} "
            f"val_rmse={val_metrics.rmse:.4f} best_epoch={best_epoch}"
        )
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_history_rows(output_dir / "training_history.csv", history)
    best_path = checkpoints_dir / "best_double_mutation_package.pt"
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written.")
    package = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(package["model_state_dict"])

    final_payload: dict[str, Any] = {
        "best_checkpoint": str(best_path),
        "best_epoch": int(package["epoch"]),
        "best_val_metrics": package["metrics"],
        "validation_metrics_path": str(
            output_dir / "validation" / "validation_set" / "phase1_best" / "metrics.json"
        ),
        "completed_at": datetime.now().isoformat(),
    }
    if test_loader is not None:
        test_loss, test_metrics, test_baseline_metrics, test_rows = evaluate_loader(
            model,
            test_loader,
            device=device,
        )
        save_prediction_rows(output_dir / "test_predictions.csv", test_rows)
        test_payload = save_split_evaluation(
            output_dir / "test_metrics" / "test_set" / "phase1_best",
            split_name="test_set",
            split_role="blind_test",
            workbook_path=args.test_xlsx,
            loss=test_loss,
            metrics=test_metrics,
            baseline_metrics=test_baseline_metrics,
            rows=test_rows,
            checkpoint_path=best_path,
            epoch=int(package["epoch"]),
        )
        final_payload["test"] = {
            "loss": test_loss,
            "double_head": test_metrics.to_dict(),
            "additive_single_baseline": test_baseline_metrics.to_dict(),
            "predictions_csv": str(output_dir / "test_predictions.csv"),
            "metrics_path": str(output_dir / "test_metrics" / "test_set" / "phase1_best" / "metrics.json"),
        }
        final_payload["test_set_metrics"] = test_payload
        save_json(output_dir / "test_metrics.json", final_payload["test"])

    save_json(output_dir / "training_summary.json", final_payload)
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
