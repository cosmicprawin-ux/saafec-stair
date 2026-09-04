#!/usr/bin/env python3
"""
train_phase1_head.py
--------------------
Phase 1 trainer for the cached-embedding SaProt stability predictor.

This stage trains only the mutation-aware stability head on cached pre-final-LayerNorm
SaProt embeddings. Fine-tuning of the backbone, final LayerNorm, and LoRA
adapters is intentionally deferred to later phases and is not implemented here.

Architecture:
    Input:  output.embeddings (pre-final-LayerNorm), cached as .pt files
    Head:   WT SaProt residue context + pooled WT protein context
            + learned WT/mutant amino-acid identity embeddings
            → residual mutation-conditioned MLP
    DDG:    ΔΔG(i,a) = raw_score(i,a) - raw_score(i,a_wt)
    Loss:   weighted masked MSE + within-protein ranking loss for training;
            unweighted masked MSE and correlations for validation/testing

Phase 1 recipe (the supervised workflow):
    Optimizer:  AdamW (β1=0.9, β2=0.999, ε=1e-8)
    LR:         1e-3 (head only)
    Schedule:   Cosine decay with 5% linear warmup
    Weight decay: 0.01 on Linear weights only (not LayerNorm)
    Grad clip:  Max norm 1.0
    Batch size: 1 protein
    Grad accumulation: 4 proteins
    Epoch cap: 100
    Early stop: patience 10 on selected validation checkpoint metric with min_delta 0.005
    Reproducibility: seed 1337, deterministic cuDNN settings

Pipeline
--------
    1. cache/generate_saprot_structure_aware_cache.py caches per-residue embeddings from PDB files
    2. This script trains the head on cached embeddings + MegaScale labels
       from the training-set CSV
    3. inference/predict_ddg_matrices.py uses the trained head for later blind-set inference

Usage
-----
    python code/training/train_phase1_head.py \\
        --mutations-csv main_training.csv \\
        --embeddings-dir ../output/embeddings/by_protein \\
        --splits-csv splits.csv \\
        --output-dir ../output/training

    # Minimal workflow-aligned invocation:
    python code/training/train_phase1_head.py --mutations-csv main_training.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from models.stability_head import (
    HEAD_DROPOUT_BY_PHASE,
    MutationAwareStabilityHead,
    predict_ddg_from_head,
)
from models.stability_loss import (
    MaskedMSELoss,
    add_composite_loss_args,
    composite_loss_from_args,
)
from core.megascale_dataset import MegaScaleDataset, megascale_collate_fn
from core.pipeline_config import (
    SCRIPTS_DIR,
    WORK_DIR,
    ensure_output_root,
    resolve_output_path,
    work_path_str,
)
from utils.output_error_logging import OutputErrorLogger, infer_output_base
from core.stability_metrics import EvalResult, derive_val_mse_rmse, evaluate
from core.checkpointing import (
    build_phase_package,
    capture_named_optimizer_state,
)
from training.megascale_test_eval import (
    build_mutation_prediction_rows_for_protein,
    print_megascale_test_metrics_payload,
    resolve_test_dataset_specs,
    save_megascale_test_metrics,
    save_prediction_reuse_subset_metrics,
    write_megascale_test_phase_summary,
)


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


CHECKPOINT_SELECTION_METRICS = ("global_pearson", "validation_composite")


def checkpoint_selection_score(result: EvalResult, metric_name: str) -> float:
    if metric_name == "global_pearson":
        return result.global_pearson
    if metric_name == "validation_composite":
        return result.dev_score
    raise ValueError(f"Unsupported checkpoint selection metric: {metric_name}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the supervised deep stability head on cached SaProt embeddings."
    )

    # Data — two mutually exclusive routes:
    #   (a) --train-xlsx + --val-xlsx  (the workbook-based workflow, one workbook per split)
    #   (b) --mutations-csv + --splits-csv  (legacy single-table route)
    parser.add_argument(
        "--train-xlsx",
        default=None,
        help=(
            "Path to the MegaScale train workbook (sheet refined_sorted_clean). "
            "Relative paths are resolved under the configured `../../data/` directory. "
            "Mutually exclusive with --mutations-csv."
        ),
    )
    parser.add_argument(
        "--val-xlsx",
        default=None,
        help="Path to the MegaScale validation workbook (sheet refined_sorted_clean).",
    )
    parser.add_argument(
        "--test-xlsx",
        action="append",
        default=None,
        help=(
            "Held-out test workbook(s), as path.xlsx or name=path.xlsx. "
            "May be repeated or comma-separated. Default: auto-discover every "
            "testing *_duplicate_homology_filtered.xlsx workbook."
        ),
    )
    parser.add_argument(
        "--test-structure-set",
        default="colabfold",
        choices=[
            "colabfold",
            "colabfold_rank45_subset",
            "modelled_rank45",
            "modeled_rank45",
        ],
        help=(
            "Structure set to auto-discover for held-out test evaluation. "
            "modelled_rank45 uses refined_sorted_modeled_only."
        ),
    )
    parser.add_argument(
        "--mutations-csv",
        default=None,
        help=(
            "Legacy: single mutations table (CSV or XLSX) covering every split. "
            "Used together with --splits-csv. Prefer --train-xlsx / --val-xlsx."
        ),
    )
    parser.add_argument(
        "--splits-csv",
        default=None,
        help="Legacy: splits CSV (protein_name, split). Only used with --mutations-csv.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default="output/embeddings/by_protein",
        help="Path to by_protein/ directory with cached SaProt embeddings.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="output/training",
        help="Directory for checkpoints, logs, and final weights.",
    )

    # Supervised training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Proteins per optimizer micro-step. Workflow-aligned default is 1.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps. Workflow-aligned default is 4 proteins.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate for the head.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.05,
                        help="Fraction of total steps for linear LR warmup.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val improvement).")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Minimum required improvement in selected checkpoint metric.")
    parser.add_argument(
        "--checkpoint-selection-metric",
        default="validation_composite",
        choices=CHECKPOINT_SELECTION_METRICS,
        help=(
            "Validation metric used for best-checkpoint selection. "
            "global_pearson pools all validation examples; validation_composite uses the "
            "configured MegaScale/external validation Pearson mixture."
        ),
    )
    parser.add_argument(
        "--dev-score-weight-megascale",
        type=float,
        default=0.5,
        help="validation-score weight for MegaScale validation Pearson.",
    )
    parser.add_argument(
        "--dev-score-weight-external",
        type=float,
        default=0.5,
        help="validation-score weight for external homolog validation mean Pearson.",
    )
    add_composite_loss_args(parser)

    # Device
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # Misc
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers (0 = main process).")
    parser.add_argument(
        "--skip-megascale-test-eval",
        action="store_true",
        help="Skip the one-time held-out test evaluation for the phase-best checkpoint.",
    )

    args = parser.parse_args()
    if args.dev_score_weight_megascale < 0 or args.dev_score_weight_external < 0:
        parser.error("validation-score weights must be non-negative.")
    if args.dev_score_weight_megascale + args.dev_score_weight_external <= 0:
        parser.error("At least one validation-score weight must be positive.")
    return args


def set_reproducibility(seed: int) -> None:
    """Apply the deterministic Phase 1 training settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup for supervised training
# ---------------------------------------------------------------------------

def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    head: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward pass + loss for one batch.  Returns scalar loss (unscaled by grad_accum).
    """
    embeddings = batch["embeddings"].to(device)  # (B, L_max, d_model)
    ca_coordinates = batch.get("ca_coordinates")
    if ca_coordinates is not None:
        ca_coordinates = ca_coordinates.to(device)
    targets = batch["targets"].to(device)         # (B, L_max, 20)
    masks = batch["masks"].to(device)             # (B, L_max, 20)
    sequences = batch["sequences"]

    ddg_pred = predict_ddg_from_head(
        head,
        embeddings,
        sequences,
        lengths=batch["lengths"],
        ca_coordinates=ca_coordinates,
    )

    loss = criterion(ddg_pred, targets, masks)
    return loss


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    head: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    save_predictions: bool = False,
    mutation_prediction_rows: list[dict] | None = None,
    dev_score_weight_megascale: float = 0.3,
    dev_score_weight_external_homolog: float = 0.7,
) -> tuple[float, EvalResult, dict[str, torch.Tensor] | None]:
    """
    Run evaluation on an entire split.

    Parameters
    ----------
    save_predictions : bool
        If True, also return a dict mapping protein_name -> DDG tensor (L, 20).

    Returns (avg_loss, EvalResult, predictions_dict_or_None).
    """
    head.eval()
    total_loss = 0.0
    n_batches = 0

    all_ddg_pred: list[torch.Tensor] = []
    all_ddg_true: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_names: list[str] = []
    all_source_databases: list[str] = []
    predictions: dict[str, torch.Tensor] = {} if save_predictions else {}

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        ca_coordinates = batch.get("ca_coordinates")
        if ca_coordinates is not None:
            ca_coordinates = ca_coordinates.to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        sequences = batch["sequences"]
        names = batch["names"]
        source_databases = batch.get("source_databases", ["unknown"] * len(names))
        mutation_resolutions = batch.get("mutation_resolutions", [None] * len(names))

        ddg_pred = predict_ddg_from_head(
            head,
            embeddings,
            sequences,
            lengths=batch["lengths"],
            ca_coordinates=ca_coordinates,
        )

        loss = criterion(ddg_pred, targets, masks)
        total_loss += loss.item()
        n_batches += 1

        # Collect per-protein predictions for metrics
        B = ddg_pred.shape[0]
        for i in range(B):
            L = batch["lengths"][i]
            all_ddg_pred.append(ddg_pred[i, :L].cpu())
            all_ddg_true.append(targets[i, :L].cpu())
            all_masks.append(masks[i, :L].cpu())
            all_names.append(names[i])
            all_source_databases.append(source_databases[i])

            if save_predictions:
                predictions[names[i]] = ddg_pred[i, :L].cpu()
            if mutation_prediction_rows is not None:
                mutation_prediction_rows.extend(
                    build_mutation_prediction_rows_for_protein(
                        ddg_pred=ddg_pred[i, :L],
                        target=targets[i, :L],
                        mask=masks[i, :L],
                        wt_sequence=sequences[i],
                        protein_name=names[i],
                        mutation_resolution=mutation_resolutions[i],
                    )
                )

    avg_loss = total_loss / max(n_batches, 1)
    eval_result = evaluate(
        all_ddg_pred,
        all_ddg_true,
        all_masks,
        all_names,
        all_source_databases,
        dev_score_weight_megascale=dev_score_weight_megascale,
        dev_score_weight_external_homolog=dev_score_weight_external_homolog,
    )

    head.train()
    return avg_loss, eval_result, predictions if save_predictions else None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = ensure_output_root(args.output_dir)
    error_logger = OutputErrorLogger(
        "step1_phase1",
        infer_output_base(output_dir, WORK_DIR),
    )

    try:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        embeddings_dir = resolve_output_path(args.embeddings_dir)

        # Resolve the dataset route (XLSX-per-split vs legacy CSV+splits).
        use_xlsx_splits = args.train_xlsx is not None or args.val_xlsx is not None
        if use_xlsx_splits:
            if args.train_xlsx is None or args.val_xlsx is None:
                raise SystemExit(
                    "--train-xlsx and --val-xlsx must be provided together."
                )
            if args.mutations_csv is not None or args.splits_csv is not None:
                raise SystemExit(
                    "XLSX splits route is exclusive: do not pass --mutations-csv / --splits-csv."
                )
            train_table: str = args.train_xlsx
            val_table: str = args.val_xlsx
            legacy_splits_csv: str | None = None
        else:
            if args.mutations_csv is None:
                raise SystemExit(
                    "Provide either --train-xlsx + --val-xlsx (the workbook-based workflow) or "
                    "--mutations-csv [+ --splits-csv] (legacy)."
                )
            train_table = args.mutations_csv
            val_table = args.mutations_csv
            legacy_splits_csv = args.splits_csv

        error_logger.write_run_status(
            "started",
            summary={
                "train_table": train_table,
                "val_table": val_table,
                "splits_csv": legacy_splits_csv,
                "embeddings_dir": work_path_str(embeddings_dir),
                "output_dir": work_path_str(output_dir),
                "seed": args.seed,
                "device_arg": args.device,
                "dataset_route": "xlsx_per_split" if use_xlsx_splits else "csv_plus_splits",
            },
        )
        error_logger.info(
            "PHASE1_START",
            "Starting Phase 1 head training.",
            context={
                "train_table": train_table,
                "val_table": val_table,
                "splits_csv": legacy_splits_csv,
                "embeddings_dir": work_path_str(embeddings_dir),
                "output_dir": work_path_str(output_dir),
                "run_status_path": work_path_str(error_logger.run_status_path),
            },
        )

        # --- Seed / reproducibility ---
        error_logger.info(
            "REPRODUCIBILITY",
            "Applying deterministic seed settings.",
            context={"seed": args.seed},
        )
        set_reproducibility(args.seed)

        # --- Device ---
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda" and not torch.cuda.is_available():
            error_logger.warning(
                "CUDA_UNAVAILABLE",
                "CUDA was requested but is unavailable; falling back to CPU.",
            )
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        print(f"Device: {device}")
        error_logger.info(
            "DEVICE_SELECTED",
            "Resolved training device.",
            context={"device": str(device), "cuda_available": torch.cuda.is_available()},
        )

        if not embeddings_dir.is_dir():
            error_logger.error(
                "EMBEDDINGS_DIR_MISSING",
                "Cached embedding directory does not exist. Run the cache-generation script first.",
                context={"embeddings_dir": work_path_str(embeddings_dir)},
            )
            error_logger.write_run_status(
                "failed",
                summary={
                    "failure_reason": "embeddings_dir_missing",
                    "embeddings_dir": work_path_str(embeddings_dir),
                },
            )
            sys.exit(1)

        # --- Datasets ---
        error_logger.info(
            "DATASET_BUILD",
            "Building Phase 1 train/validation datasets from cached embeddings.",
            context={
                "train_table": train_table,
                "val_table": val_table,
                "embeddings_dir": work_path_str(embeddings_dir),
                "splits_csv": legacy_splits_csv,
            },
        )
        train_ds = MegaScaleDataset(
            mutations_table=train_table,
            embeddings_dir=embeddings_dir,
            split="train",
            splits_csv=legacy_splits_csv,
        )
        val_ds = MegaScaleDataset(
            mutations_table=val_table,
            embeddings_dir=embeddings_dir,
            split="val" if not use_xlsx_splits else "val",
            splits_csv=legacy_splits_csv,
        )
        test_specs = (
            []
            if args.skip_megascale_test_eval
            else resolve_test_dataset_specs(
                args.test_xlsx,
                structure_set=args.test_structure_set,
            )
        )
        test_items: list[dict[str, Any]] = []
        if not args.skip_megascale_test_eval:
            for test_spec in test_specs:
                if test_spec.prediction_source_name is not None:
                    continue
                test_items.append(
                    {
                        "spec": test_spec,
                        "dataset": MegaScaleDataset(
                            mutations_table=test_spec.table,
                            embeddings_dir=embeddings_dir,
                            split="test",
                            splits_csv=None,
                            workbook_is_split=True,
                            xlsx_sheet_name=test_spec.sheet_name,
                        ),
                    }
                )

        if len(train_ds) == 0:
            error_logger.error(
                "TRAIN_DATASET_EMPTY",
                "Training set is empty. Check mutations table, embeddings dir, and splits.",
                context={
                    "train_table": train_table,
                    "embeddings_dir": work_path_str(embeddings_dir),
                    "splits_csv": legacy_splits_csv,
                },
            )
            error_logger.write_run_status(
                "failed",
                summary={
                    "failure_reason": "train_dataset_empty",
                    "train_table": train_table,
                    "embeddings_dir": work_path_str(embeddings_dir),
                    "splits_csv": legacy_splits_csv,
                },
            )
            sys.exit(1)
        if len(val_ds) == 0:
            error_logger.warning(
                "VALIDATION_DATASET_EMPTY",
                "Validation set is empty. Early stopping will be disabled.",
            )
        for test_item in test_items:
            test_spec = test_item["spec"]
            test_ds = test_item["dataset"]
            if len(test_ds) == 0:
                error_logger.error(
                    "TEST_DATASET_EMPTY",
                    "Held-out test set is empty. Check the test workbook and cached embeddings.",
                    context={
                        "test_dataset": test_spec.name,
                        "test_table": test_spec.table,
                        "embeddings_dir": work_path_str(embeddings_dir),
                    },
                )
                error_logger.write_run_status(
                    "failed",
                    summary={
                        "failure_reason": "test_dataset_empty",
                        "test_dataset": test_spec.name,
                        "test_table": test_spec.table,
                        "embeddings_dir": work_path_str(embeddings_dir),
                    },
                )
                sys.exit(1)
        error_logger.info(
            "DATASET_READY",
            "Datasets loaded successfully.",
            context={
                "train_proteins": len(train_ds),
                "val_proteins": len(val_ds),
                "test_datasets": [
                    {
                        "name": item["spec"].name,
                        "table": item["spec"].table,
                        "proteins": len(item["dataset"]),
                    }
                    for item in test_items
                ],
            },
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=megascale_collate_fn,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=megascale_collate_fn,
            num_workers=args.num_workers,
            drop_last=False,
        )
        for test_item in test_items:
            test_item["loader"] = DataLoader(
                test_item["dataset"],
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=megascale_collate_fn,
                num_workers=args.num_workers,
                drop_last=False,
            )

        # --- Model ---
        first_embedding = train_ds[0].embedding
        if first_embedding.ndim != 2:
            raise ValueError(
                f"Expected cached embeddings to have shape (L, D), got {tuple(first_embedding.shape)}"
            )
        embedding_dim = int(first_embedding.shape[1])
        head = MutationAwareStabilityHead(
            d_model=embedding_dim,
            dropout=HEAD_DROPOUT_BY_PHASE[1],
        ).to(device)
        train_criterion = composite_loss_from_args(args)
        eval_criterion = MaskedMSELoss()

        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        print(f"{head.__class__.__name__} trainable parameters: {trainable_params:,}")
        error_logger.info(
            "MODEL_READY",
            "Phase 1 mutation-aware stability head initialised.",
            context={
                "trainable_params": trainable_params,
                "embedding_dim": embedding_dim,
                "head_dropout": HEAD_DROPOUT_BY_PHASE[1],
                "head_architecture": getattr(head, "architecture_name", head.__class__.__name__),
                "local_contact_top_k": getattr(head, "local_contact_top_k", None),
                "local_contact_cutoff": getattr(head, "local_contact_cutoff", None),
                "local_contact_distance_scale": getattr(head, "local_contact_distance_scale", None),
                "training_loss": train_criterion.to_config(),
                "validation_loss": "MaskedMSELoss",
            },
        )

        # --- Optimizer for supervised training ---
        # Weight decay on Linear weights only, not LayerNorm
        decay_params = []
        no_decay_params = []
        for name, param in head.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # --- LR schedule ---
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_frac)
        scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
        print(
            f"Schedule: {total_steps} total steps, {warmup_steps} warmup steps, "
            f"{steps_per_epoch} steps/epoch"
        )

        # --- Training config summary ---
        effective_batch = args.batch_size * args.grad_accum
        config = {
            "workflow": "phase1_cached_embeddings",
            "phase": 1,
            "head_architecture": getattr(head, "architecture_name", head.__class__.__name__),
            "head_d_model": embedding_dim,
            "head_params": trainable_params,
            "head_dropout": HEAD_DROPOUT_BY_PHASE[1],
            "local_contact_top_k": getattr(head, "local_contact_top_k", None),
            "local_contact_cutoff": getattr(head, "local_contact_cutoff", None),
            "local_contact_distance_scale": getattr(head, "local_contact_distance_scale", None),
            "training_objective": train_criterion.to_config(),
            "validation_loss": "MaskedMSELoss",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size_per_device": args.batch_size,
            "grad_accumulation": args.grad_accum,
            "effective_batch_size": effective_batch,
            "warmup_frac": args.warmup_frac,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "device": str(device),
            "train_proteins": len(train_ds),
            "val_proteins": len(val_ds),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "report_aligned_defaults": {
                "batch_size": 1,
                "grad_accum": 4,
                "lr": 1e-3,
                "epochs": 100,
                "patience": 10,
                "min_delta": 0.005,
                "head_dropout": 0.10,
            },
            "train_table": train_table,
            "val_table": val_table,
                    "test_tables": [
                        {
                            "name": item["spec"].name,
                            "table": item["spec"].table,
                            "xlsx_sheet": item["spec"].sheet_name,
                            "structure_set": item["spec"].structure_set,
                            "proteins": len(item["dataset"]),
                        }
                        for item in test_items
            ],
            "embeddings_dir": args.embeddings_dir,
            "splits_csv": legacy_splits_csv,
            "dataset_route": "xlsx_per_split" if use_xlsx_splits else "csv_plus_splits",
            "started_at": datetime.now().isoformat(),
            "error_log_path": work_path_str(error_logger.script_log_path),
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "dev_score_weight_megascale": args.dev_score_weight_megascale,
            "dev_score_weight_external_homolog": args.dev_score_weight_external,
        }
        (output_dir / "train_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        print(f"Effective batch size: {effective_batch} proteins")
        error_logger.info(
            "TRAINING_CONFIG_WRITTEN",
            "Saved Phase 1 training configuration.",
            context={"train_config_path": work_path_str(output_dir / "train_config.json")},
        )

        # --- Training loop ---
        best_val_global_spearman = float("-inf")
        best_val_global_pearson = float("-inf")
        best_val_dev_score = float("-inf")
        best_checkpoint_score = float("-inf")
        best_checkpoint_val_global_pearson = float("-inf")
        best_checkpoint_val_dev_score = float("-inf")
        patience_counter = 0
        history: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            epoch_start = perf_counter()
            head.train()
            optimizer.zero_grad()

            epoch_loss = 0.0
            micro_steps = 0
            global_step = 0

            for batch_idx, batch in enumerate(train_loader):
                loss = train_step(batch, head, train_criterion, device)
                scaled_loss = loss / args.grad_accum
                scaled_loss.backward()

                epoch_loss += loss.item()
                micro_steps += 1

                if micro_steps % args.grad_accum == 0 or batch_idx == len(train_loader) - 1:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            avg_train_loss = epoch_loss / max(micro_steps, 1)
            epoch_time = perf_counter() - epoch_start

            val_loss = float("nan")
            val_result = EvalResult()
            if len(val_ds) > 0:
                val_loss, val_result, _ = evaluate_split(
                    val_loader,
                    head,
                    eval_criterion,
                    device,
                    dev_score_weight_megascale=args.dev_score_weight_megascale,
                    dev_score_weight_external_homolog=args.dev_score_weight_external,
                )
            val_mse, val_rmse = derive_val_mse_rmse(val_loss)

            val_metrics = {f"val_{k}": v for k, v in val_result.to_dict().items()}
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_log = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_mse": avg_train_loss,
                **val_metrics,
                "checkpoint_selection_metric": args.checkpoint_selection_metric,
                "checkpoint_selection_score": checkpoint_selection_score(
                    val_result,
                    args.checkpoint_selection_metric,
                ),
                "val_mse": val_mse,
                "val_rmse": val_rmse,
                "lr": current_lr,
                "epoch_time_s": epoch_time,
            }
            history.append(epoch_log)

            val_spearman = val_result.global_spearman
            val_pearson = val_result.global_pearson
            val_dev_score = val_result.dev_score
            checkpoint_score = checkpoint_selection_score(
                val_result,
                args.checkpoint_selection_metric,
            )
            if val_spearman > best_val_global_spearman:
                best_val_global_spearman = val_spearman
            if val_result.global_pearson > best_val_global_pearson:
                best_val_global_pearson = val_result.global_pearson
            if val_result.dev_score > best_val_dev_score:
                best_val_dev_score = val_result.dev_score
            improved = ""
            if (
                checkpoint_score == checkpoint_score and
                checkpoint_score > best_checkpoint_score + args.min_delta
            ):
                best_checkpoint_score = checkpoint_score
                best_checkpoint_val_global_pearson = val_pearson
                best_checkpoint_val_dev_score = val_dev_score
                patience_counter = 0
                improved = " *"
                torch.save(head.state_dict(), ckpt_dir / "best_head.pt")
                best_phase1_package = build_phase_package(
                    phase=1,
                    epoch=epoch,
                    head=head,
                    model=None,
                    val_mse=val_loss,
                    val_metrics=val_result.to_full_dict(),
                    phase_spec=None,
                    train_config=config,
                    head_optimizer_named_state=capture_named_optimizer_state(optimizer, head),
                )
                torch.save(best_phase1_package, ckpt_dir / "phase1_best_package.pt")
                (output_dir / "best_val_results.json").write_text(
                    json.dumps({
                        "phase": 1,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_mse": val_mse,
                        "val_rmse": val_rmse,
                        "checkpoint_selection_metric": args.checkpoint_selection_metric,
                        "checkpoint_selection_score": checkpoint_score,
                        **val_result.to_full_dict(),
                    }, indent=2),
                    encoding="utf-8",
                )
                save_csv_rows(
                    output_dir / "best_metrics_summary.csv",
                    [
                        "phase",
                        "epoch",
                        "val_mse",
                        "val_rmse",
                        "val_mae",
                        "median_per_protein_spearman",
                        "median_per_protein_pearson",
                        "val_global_spearman",
                        "val_global_pearson",
                        "val_dev_score",
                        "checkpoint_selection_metric",
                        "checkpoint_selection_score",
                        "val_megascale_validation_pearson",
                        "val_external_homolog_validation_mean_pearson",
                        "stabilizing_ppv",
                        "n_proteins",
                        "n_mutations",
                        "best_checkpoint",
                    ],
                    [
                        {
                            "phase": 1,
                            "epoch": epoch,
                            "val_mse": val_mse,
                            "val_rmse": val_rmse,
                            "val_mae": val_result.mae,
                            "median_per_protein_spearman": val_result.median_per_protein_spearman,
                            "median_per_protein_pearson": val_result.median_per_protein_pearson,
                            "val_global_spearman": val_result.global_spearman,
                            "val_global_pearson": val_result.global_pearson,
                            "val_dev_score": val_result.dev_score,
                            "checkpoint_selection_metric": args.checkpoint_selection_metric,
                            "checkpoint_selection_score": checkpoint_score,
                            "val_megascale_validation_pearson": val_result.megascale_validation_pearson,
                            "val_external_homolog_validation_mean_pearson": val_result.external_homolog_validation_mean_pearson,
                            "stabilizing_ppv": val_result.stabilizing_ppv,
                            "n_proteins": val_result.n_proteins,
                            "n_mutations": val_result.n_mutations,
                            "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                        }
                    ],
                )
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={avg_train_loss:.5f}  val_loss={val_loss:.5f} | "
                f"val_g_spearman={val_result.global_spearman:.4f}  "
                f"val_g_pearson={val_result.global_pearson:.4f}  "
                f"dev_score={val_result.dev_score:.4f}  "
                f"rmse={val_result.rmse:.4f}  mae={val_result.mae:.4f}  "
                f"ppv={val_result.stabilizing_ppv:.4f} | "
                f"lr={current_lr:.2e}  time={epoch_time:.1f}s{improved}"
            )

            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save(head.state_dict(), ckpt_dir / f"head_epoch{epoch:03d}.pt")

            if len(val_ds) > 0 and patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs). "
                    f"Best checkpoint {args.checkpoint_selection_metric}: {best_checkpoint_score:.4f} "
                    f"(Best val global Spearman overall: {best_val_global_spearman:.4f})"
                )
                break

        # --- Save final outputs ---
        if not (ckpt_dir / "best_head.pt").is_file():
            torch.save(head.state_dict(), ckpt_dir / "best_head.pt")

        # Final head weights
        torch.save(head.state_dict(), ckpt_dir / "final_head.pt")
        final_phase1_package = build_phase_package(
            phase=1,
            epoch=len(history),
            head=head,
            model=None,
            val_mse=history[-1]["val_mse"] if history else float("nan"),
            val_metrics=(
                {k.removeprefix("val_"): v for k, v in history[-1].items() if k.startswith("val_")}
                if history else EvalResult().to_full_dict()
            ),
            phase_spec=None,
            train_config=config,
            head_optimizer_named_state=capture_named_optimizer_state(optimizer, head),
        )
        torch.save(final_phase1_package, ckpt_dir / "phase1_final_package.pt")
        if not (ckpt_dir / "phase1_best_package.pt").is_file():
            torch.save(final_phase1_package, ckpt_dir / "phase1_best_package.pt")

        phase1_test_metrics_payloads: list[dict[str, Any]] = []
        phase1_test_output_dirs: list[Path] = []
        phase1_test_metric_paths: list[Path] = []
        if test_items:
            best_checkpoint_path = ckpt_dir / "best_head.pt"
            best_package_path = ckpt_dir / "phase1_best_package.pt"
            best_phase1_eval_package = torch.load(
                best_package_path,
                map_location="cpu",
                weights_only=False,
            )
            best_head_for_test = MutationAwareStabilityHead(
                d_model=int(best_phase1_eval_package.get("head_d_model", embedding_dim)),
                dropout=HEAD_DROPOUT_BY_PHASE[1],
            ).to(device)
            best_head_for_test.load_state_dict(
                torch.load(best_checkpoint_path, map_location=device, weights_only=True)
            )
            prediction_rows_by_name: dict[str, list[dict[str, Any]]] = {}
            for test_item in test_items:
                test_spec = test_item["spec"]
                mutation_prediction_rows: list[dict] = []
                test_loss, test_result, _ = evaluate_split(
                    test_item["loader"],
                    best_head_for_test,
                    eval_criterion,
                    device,
                    mutation_prediction_rows=mutation_prediction_rows,
                    dev_score_weight_megascale=args.dev_score_weight_megascale,
                    dev_score_weight_external_homolog=args.dev_score_weight_external,
                )
                test_output_dir = output_dir / "test_metrics" / test_spec.name / "phase1_best"
                test_metrics_payload = save_megascale_test_metrics(
                    output_dir=test_output_dir,
                    phase=1,
                    phase_name="phase1_head_only",
                    epoch=int(best_phase1_eval_package["epoch"]),
                    checkpoint_path=best_checkpoint_path,
                    test_table=test_spec.table,
                    test_loss=test_loss,
                        test_result=test_result,
                        test_name=test_spec.name,
                        test_xlsx_sheet=test_spec.sheet_name,
                        test_structure_set=test_spec.structure_set,
                        mutation_prediction_rows=mutation_prediction_rows,
                )
                phase1_test_metrics_payloads.append(test_metrics_payload)
                phase1_test_output_dirs.append(test_output_dir)
                phase1_test_metric_paths.append(test_output_dir / "metrics.json")
                prediction_rows_by_name[test_spec.name] = mutation_prediction_rows
            for test_spec in test_specs:
                if test_spec.prediction_source_name is None:
                    continue
                source_rows = prediction_rows_by_name.get(test_spec.prediction_source_name)
                if source_rows is None:
                    raise RuntimeError(
                        f"Test dataset {test_spec.name} reuses predictions from "
                        f"{test_spec.prediction_source_name}, but that source was not evaluated."
                    )
                test_output_dir = output_dir / "test_metrics" / test_spec.name / "phase1_best"
                test_metrics_payload = save_prediction_reuse_subset_metrics(
                    output_dir=test_output_dir,
                    phase=1,
                    phase_name="phase1_head_only",
                    epoch=int(best_phase1_eval_package["epoch"]),
                    checkpoint_path=best_checkpoint_path,
                    subset_spec=test_spec,
                    source_rows=source_rows,
                )
                phase1_test_metrics_payloads.append(test_metrics_payload)
                phase1_test_output_dirs.append(test_output_dir)
                phase1_test_metric_paths.append(test_output_dir / "metrics.json")
            write_megascale_test_phase_summary(
                output_dir=output_dir / "test_metrics",
                metric_paths=phase1_test_metric_paths,
            )
        if not (output_dir / "best_val_results.json").is_file():
            fallback_metrics = {
                k.removeprefix("val_"): v for k, v in history[-1].items() if k.startswith("val_")
            } if history else EvalResult().to_full_dict()
            (output_dir / "best_val_results.json").write_text(
                json.dumps({
                    "phase": 1,
                    "epoch": len(history),
                    "val_loss": history[-1]["val_loss"] if history else float("nan"),
                    "val_mse": history[-1]["val_mse"] if history else float("nan"),
                    "val_rmse": history[-1]["val_rmse"] if history else float("nan"),
                    "checkpoint_selection_metric": args.checkpoint_selection_metric,
                    "checkpoint_selection_score": (
                        history[-1]["checkpoint_selection_score"] if history else float("nan")
                    ),
                    **fallback_metrics,
                }, indent=2),
                encoding="utf-8",
            )
        if not (output_dir / "best_metrics_summary.csv").is_file() and history:
            save_csv_rows(
                output_dir / "best_metrics_summary.csv",
                [
                    "phase",
                    "epoch",
                    "val_mse",
                    "val_rmse",
                    "val_mae",
                    "median_per_protein_spearman",
                    "median_per_protein_pearson",
                    "val_global_spearman",
                    "val_global_pearson",
                    "val_dev_score",
                    "checkpoint_selection_metric",
                    "checkpoint_selection_score",
                    "val_megascale_validation_pearson",
                    "val_external_homolog_validation_mean_pearson",
                    "stabilizing_ppv",
                    "n_proteins",
                    "n_mutations",
                    "best_checkpoint",
                ],
                [
                    {
                        "phase": 1,
                        "epoch": len(history),
                        "val_mse": history[-1]["val_mse"],
                        "val_rmse": history[-1]["val_rmse"],
                        "val_mae": history[-1]["val_mae"],
                        "median_per_protein_spearman": history[-1]["val_median_per_protein_spearman"],
                        "median_per_protein_pearson": history[-1]["val_median_per_protein_pearson"],
                        "val_global_spearman": history[-1]["val_global_spearman"],
                        "val_global_pearson": history[-1]["val_global_pearson"],
                        "val_dev_score": history[-1]["val_dev_score"],
                        "checkpoint_selection_metric": history[-1]["checkpoint_selection_metric"],
                        "checkpoint_selection_score": history[-1]["checkpoint_selection_score"],
                        "val_megascale_validation_pearson": history[-1]["val_megascale_validation_pearson"],
                        "val_external_homolog_validation_mean_pearson": history[-1]["val_external_homolog_validation_mean_pearson"],
                        "stabilizing_ppv": history[-1]["val_stabilizing_ppv"],
                        "n_proteins": history[-1]["val_n_proteins"],
                        "n_mutations": history[-1]["val_n_mutations"],
                        "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                    }
                ],
            )

        # Training history
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        save_csv_rows(
            output_dir / "training_history_metrics.csv",
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_mse",
                "val_mse",
                "val_rmse",
                "val_mae",
                "val_median_per_protein_spearman",
                "val_median_per_protein_pearson",
                "val_global_spearman",
                "val_global_pearson",
                "val_dev_score",
                "checkpoint_selection_metric",
                "checkpoint_selection_score",
                "val_megascale_validation_pearson",
                "val_external_homolog_validation_mean_pearson",
                "val_stabilizing_ppv",
                "val_n_proteins",
                "val_n_mutations",
                "lr",
                "epoch_time_s",
            ],
            history,
        )

        # Final summary
        summary = {
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "dev_score_weight_megascale": args.dev_score_weight_megascale,
            "dev_score_weight_external_homolog": args.dev_score_weight_external,
            "best_checkpoint_score": best_checkpoint_score,
            "best_checkpoint_val_global_pearson": best_checkpoint_val_global_pearson,
            "best_checkpoint_val_dev_score": best_checkpoint_val_dev_score,
            "best_val_global_spearman": best_val_global_spearman,
            "best_val_global_pearson": best_val_global_pearson,
            "best_val_dev_score": best_val_dev_score,
            "total_epochs_run": len(history),
            "early_stopped": patience_counter >= args.patience,
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "final_val_mse": history[-1]["val_mse"] if history else None,
            "final_val_rmse": history[-1]["val_rmse"] if history else None,
            "best_checkpoint": str(ckpt_dir / "best_head.pt"),
            "final_checkpoint": str(ckpt_dir / "final_head.pt"),
            "best_phase_package": str(ckpt_dir / "phase1_best_package.pt"),
            "final_phase_package": str(ckpt_dir / "phase1_final_package.pt"),
            "test_metrics": phase1_test_metrics_payloads,
            "finished_at": datetime.now().isoformat(),
            "error_log_path": work_path_str(error_logger.script_log_path),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        error_logger.write_run_status(
            "completed",
            summary={
                "output_dir": work_path_str(output_dir),
                "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                "final_checkpoint": work_path_str(ckpt_dir / "final_head.pt"),
                "epochs_run": len(history),
                "best_val_global_spearman": best_val_global_spearman,
                "best_val_global_pearson": best_val_global_pearson,
                "best_val_dev_score": best_val_dev_score,
                "best_checkpoint_score": best_checkpoint_score,
                "best_checkpoint_val_global_pearson": best_checkpoint_val_global_pearson,
                "best_checkpoint_val_dev_score": best_checkpoint_val_dev_score,
            },
        )
        error_logger.info(
            "PHASE1_COMPLETE",
            "Phase 1 training completed successfully.",
            context={
                "output_dir": work_path_str(output_dir),
                "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                "run_status_path": work_path_str(error_logger.run_status_path),
            },
        )

        print(f"\nTraining complete.")
        print(f"  Best checkpoint {args.checkpoint_selection_metric}: {best_checkpoint_score:.4f}")
        print(f"  Best checkpoint val dev score: {best_checkpoint_val_dev_score:.4f}")
        print(f"  Best val dev score overall: {best_val_dev_score:.4f}")
        print(f"  Best val global Pearson overall: {best_val_global_pearson:.4f}")
        print(f"  Best val global Spearman overall: {best_val_global_spearman:.4f}")
        print(f"  Best checkpoint: {ckpt_dir / 'best_head.pt'}")
        print(f"  Outputs: {output_dir}")

        for phase1_test_metrics_payload, phase1_test_output_dir in zip(
            phase1_test_metrics_payloads,
            phase1_test_output_dirs,
        ):
            print_megascale_test_metrics_payload(
                phase1_test_metrics_payload,
                output_dir=phase1_test_output_dir,
            )
    except Exception as exc:
        error_logger.exception(
            "TRAINING_FATAL",
            "train_phase1_head.py failed before completing the training run.",
            exc,
            context={
                "train_xlsx": args.train_xlsx,
                "val_xlsx": args.val_xlsx,
                "mutations_csv": args.mutations_csv,
                "splits_csv": args.splits_csv,
                "embeddings_dir": args.embeddings_dir,
                "output_dir": args.output_dir,
            },
        )
        error_logger.write_run_status(
            "failed",
            summary={
                "train_xlsx": args.train_xlsx,
                "val_xlsx": args.val_xlsx,
                "mutations_csv": args.mutations_csv,
                "splits_csv": args.splits_csv,
                "embeddings_dir": args.embeddings_dir,
                "output_dir": args.output_dir,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
