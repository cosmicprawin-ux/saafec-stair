#!/usr/bin/env python3
"""Train the intrinsic SaProt + ProteinMPNN fusion stability head."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.checkpointing import capture_named_optimizer_state  # noqa: E402
from core.pipeline_config import WORK_DIR, ensure_output_root, resolve_output_path, work_path_str  # noqa: E402
from core.saprot_proteinmpnn_dataset import (  # noqa: E402
    SaProtProteinMPNNDataset,
    saprot_proteinmpnn_collate_fn,
)
from core.stability_metrics import EvalResult, derive_val_mse_rmse, evaluate  # noqa: E402
from models.saprot_proteinmpnn_intrinsic_fusion import (  # noqa: E402
    DEFAULT_HIDDEN,
    LOCAL_CONTACT_CUTOFF_A,
    LOCAL_CONTACT_DISTANCE_SCALE_A,
    LOCAL_CONTACT_TOP_K,
    SaProtProteinMPNNIntrinsicFusionHead,
)
from models.stability_loss import (  # noqa: E402
    MaskedMSELoss,
    add_composite_loss_args,
    composite_loss_from_args,
)
from training.megascale_test_eval import (  # noqa: E402
    build_mutation_prediction_rows_for_protein,
    print_megascale_test_metrics_payload,
    resolve_test_dataset_specs,
    save_megascale_test_metrics,
    save_prediction_reuse_subset_metrics,
    write_megascale_test_phase_summary,
)
from training.train_phase1_head import (  # noqa: E402
    CHECKPOINT_SELECTION_METRICS,
    checkpoint_selection_score,
    cosine_warmup_schedule,
    save_csv_rows,
    set_reproducibility,
)
from utils.output_error_logging import OutputErrorLogger, infer_output_base  # noqa: E402


PHASE_NAME = "phase1_saprot_proteinmpnn_intrinsic_fusion"
CHECKPOINT_FORMAT_VERSION = "saprot_proteinmpnn_intrinsic_fusion_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a non-late SaProt + ProteinMPNN intrinsic fusion head."
    )
    parser.add_argument("--train-xlsx", default=None)
    parser.add_argument("--val-xlsx", default=None)
    parser.add_argument(
        "--test-xlsx",
        action="append",
        default=None,
        help=(
            "Held-out test workbook(s), as path.xlsx, name=path.xlsx, or "
            "name=path.xlsx::sheet. May be repeated. Default: auto-discover."
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
    )
    parser.add_argument("--mutations-csv", default=None)
    parser.add_argument("--splits-csv", default=None)
    parser.add_argument(
        "--saprot-embeddings-dir",
        default="output/saprot_proteinmpnn_intrinsic_fusion/embeddings/by_protein",
    )
    parser.add_argument(
        "--proteinmpnn-cache-dir",
        default="output/saprot_proteinmpnn_intrinsic_fusion/proteinmpnn_logits/by_protein",
    )
    parser.add_argument(
        "--output-dir",
        default="output/saprot_proteinmpnn_intrinsic_fusion/phase1",
    )

    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--aa-embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--local-contact-top-k", type=int, default=LOCAL_CONTACT_TOP_K)
    parser.add_argument("--local-contact-cutoff", type=float, default=LOCAL_CONTACT_CUTOFF_A)
    parser.add_argument(
        "--local-contact-distance-scale",
        type=float,
        default=LOCAL_CONTACT_DISTANCE_SCALE_A,
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument(
        "--checkpoint-selection-metric",
        default="validation_composite",
        choices=CHECKPOINT_SELECTION_METRICS,
    )
    parser.add_argument("--dev-score-weight-megascale", type=float, default=0.5)
    parser.add_argument("--dev-score-weight-external", type=float, default=0.5)
    add_composite_loss_args(parser)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-megascale-test-eval", action="store_true")

    args = parser.parse_args()
    if args.dev_score_weight_megascale < 0 or args.dev_score_weight_external < 0:
        parser.error("validation-score weights must be non-negative.")
    if args.dev_score_weight_megascale + args.dev_score_weight_external <= 0:
        parser.error("At least one validation-score weight must be positive.")
    if args.hidden_dim % args.attention_heads != 0:
        parser.error("--hidden-dim must be divisible by --attention-heads.")
    return args


def load_state_dict(path: Path, *, map_location: torch.device | str) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state).__name__}.")
    return state


def make_head(*, d_saprot: int, args: argparse.Namespace | dict[str, Any]) -> SaProtProteinMPNNIntrinsicFusionHead:
    getter = args.get if isinstance(args, dict) else lambda key, default=None: getattr(args, key, default)
    return SaProtProteinMPNNIntrinsicFusionHead(
        d_saprot=d_saprot,
        d_hidden=int(getter("hidden_dim", DEFAULT_HIDDEN)),
        aa_embed_dim=int(getter("aa_embed_dim", 64)),
        dropout=float(getter("dropout", 0.10)),
        num_attention_heads=int(getter("attention_heads", 8)),
        num_residual_blocks=int(getter("residual_blocks", 2)),
        local_contact_top_k=int(getter("local_contact_top_k", LOCAL_CONTACT_TOP_K)),
        local_contact_cutoff=float(getter("local_contact_cutoff", LOCAL_CONTACT_CUTOFF_A)),
        local_contact_distance_scale=float(
            getter("local_contact_distance_scale", LOCAL_CONTACT_DISTANCE_SCALE_A)
        ),
    )


def build_fusion_package(
    *,
    epoch: int,
    head: nn.Module,
    val_mse: float,
    val_metrics: dict[str, Any],
    train_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    val_mse, val_rmse = derive_val_mse_rmse(float(val_mse))
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "phase": 1,
        "epoch": epoch,
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "head_architecture": getattr(head, "architecture_name", head.__class__.__name__),
        "head_d_model": int(train_config["model_config"]["hidden_dim"]),
        "d_saprot": int(train_config["model_config"]["d_saprot"]),
        "proteinmpnn_feature": "frozen_inverse_folding_logits_20aa",
        "metrics": {
            **val_metrics,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
        },
        "train_config": train_config,
        "head_optimizer_named_state": capture_named_optimizer_state(optimizer, head),
        "created_at": datetime.now().isoformat(),
    }


def predict_batch(
    batch: dict[str, Any],
    head: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    saprot_embeddings = batch["saprot_embeddings"].to(device)
    proteinmpnn_logits = batch["proteinmpnn_logits"].to(device)
    proteinmpnn_mask = batch.get("proteinmpnn_masks")
    if proteinmpnn_mask is not None:
        proteinmpnn_mask = proteinmpnn_mask.to(device)
    ca_coordinates = batch.get("ca_coordinates")
    if ca_coordinates is not None:
        ca_coordinates = ca_coordinates.to(device)
    return head(
        saprot_embeddings,
        proteinmpnn_logits,
        batch["sequences"],
        lengths=batch["lengths"],
        ca_coordinates=ca_coordinates,
        proteinmpnn_mask=proteinmpnn_mask,
    )


def train_step(
    batch: dict[str, Any],
    head: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    targets = batch["targets"].to(device)
    masks = batch["masks"].to(device)
    ddg_pred = predict_batch(batch, head, device)
    return criterion(ddg_pred, targets, masks)


@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    head: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    *,
    mutation_prediction_rows: list[dict[str, Any]] | None = None,
    dev_score_weight_megascale: float = 0.5,
    dev_score_weight_external_homolog: float = 0.5,
) -> tuple[float, EvalResult]:
    head.eval()
    total_loss = 0.0
    n_batches = 0
    all_ddg_pred: list[torch.Tensor] = []
    all_ddg_true: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_names: list[str] = []
    all_source_databases: list[str] = []

    for batch in loader:
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)
        names = batch["names"]
        sequences = batch["sequences"]
        source_databases = batch.get("source_databases", ["unknown"] * len(names))
        mutation_resolutions = batch.get("mutation_resolutions", [None] * len(names))
        ddg_pred = predict_batch(batch, head, device)
        loss = criterion(ddg_pred, targets, masks)
        total_loss += float(loss.item())
        n_batches += 1

        for i in range(ddg_pred.shape[0]):
            length = int(batch["lengths"][i])
            all_ddg_pred.append(ddg_pred[i, :length].detach().cpu())
            all_ddg_true.append(targets[i, :length].detach().cpu())
            all_masks.append(masks[i, :length].detach().cpu())
            all_names.append(names[i])
            all_source_databases.append(source_databases[i])
            if mutation_prediction_rows is not None:
                mutation_prediction_rows.extend(
                    build_mutation_prediction_rows_for_protein(
                        ddg_pred=ddg_pred[i, :length],
                        target=targets[i, :length],
                        mask=masks[i, :length],
                        wt_sequence=sequences[i],
                        protein_name=names[i],
                        mutation_resolution=mutation_resolutions[i],
                    )
                )

    avg_loss = total_loss / max(n_batches, 1)
    result = evaluate(
        all_ddg_pred,
        all_ddg_true,
        all_masks,
        all_names,
        all_source_databases,
        dev_score_weight_megascale=dev_score_weight_megascale,
        dev_score_weight_external_homolog=dev_score_weight_external_homolog,
    )
    head.train()
    return avg_loss, result


def save_best_metrics_summary(
    output_dir: Path,
    *,
    epoch: int,
    val_loss: float,
    val_result: EvalResult,
    checkpoint_selection_metric: str,
    checkpoint_selection: float,
    best_checkpoint: Path,
) -> None:
    val_mse, val_rmse = derive_val_mse_rmse(val_loss)
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
                "checkpoint_selection_metric": checkpoint_selection_metric,
                "checkpoint_selection_score": checkpoint_selection,
                "val_megascale_validation_pearson": val_result.megascale_validation_pearson,
                "val_external_homolog_validation_mean_pearson": (
                    val_result.external_homolog_validation_mean_pearson
                ),
                "stabilizing_ppv": val_result.stabilizing_ppv,
                "n_proteins": val_result.n_proteins,
                "n_mutations": val_result.n_mutations,
                "best_checkpoint": work_path_str(best_checkpoint),
            }
        ],
    )


def write_training_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
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


def dataset_for_split(
    *,
    table: str,
    saprot_embeddings_dir: Path,
    proteinmpnn_cache_dir: Path,
    split: str,
    legacy_splits_csv: str | None,
    workbook_is_split: bool | None,
    xlsx_sheet_name: str = "refined_sorted_clean",
) -> SaProtProteinMPNNDataset:
    return SaProtProteinMPNNDataset(
        mutations_table=table,
        saprot_embeddings_dir=saprot_embeddings_dir,
        proteinmpnn_cache_dir=proteinmpnn_cache_dir,
        split=split,
        splits_csv=legacy_splits_csv,
        workbook_is_split=workbook_is_split,
        xlsx_sheet_name=xlsx_sheet_name,
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_root(args.output_dir)
    error_logger = OutputErrorLogger("saprot_proteinmpnn_fusion", infer_output_base(output_dir, WORK_DIR))

    try:
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        saprot_embeddings_dir = resolve_output_path(args.saprot_embeddings_dir)
        proteinmpnn_cache_dir = resolve_output_path(args.proteinmpnn_cache_dir)

        use_xlsx_splits = args.train_xlsx is not None or args.val_xlsx is not None
        if use_xlsx_splits:
            if args.train_xlsx is None or args.val_xlsx is None:
                raise SystemExit("--train-xlsx and --val-xlsx must be provided together.")
            if args.mutations_csv is not None or args.splits_csv is not None:
                raise SystemExit("Do not mix --train-xlsx/--val-xlsx with --mutations-csv/--splits-csv.")
            train_table = args.train_xlsx
            val_table = args.val_xlsx
            legacy_splits_csv = None
            workbook_is_split = True
        else:
            if args.mutations_csv is None:
                raise SystemExit("Provide either --train-xlsx + --val-xlsx or --mutations-csv.")
            train_table = args.mutations_csv
            val_table = args.mutations_csv
            legacy_splits_csv = args.splits_csv
            workbook_is_split = None

        error_logger.write_run_status(
            "started",
            summary={
                "train_table": train_table,
                "val_table": val_table,
                "saprot_embeddings_dir": work_path_str(saprot_embeddings_dir),
                "proteinmpnn_cache_dir": work_path_str(proteinmpnn_cache_dir),
                "output_dir": work_path_str(output_dir),
                "seed": args.seed,
            },
        )

        set_reproducibility(args.seed)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        print(f"Device: {device}")

        if not saprot_embeddings_dir.is_dir():
            raise FileNotFoundError(f"SaProt embeddings directory missing: {saprot_embeddings_dir}")
        if not proteinmpnn_cache_dir.is_dir():
            raise FileNotFoundError(f"ProteinMPNN logits cache directory missing: {proteinmpnn_cache_dir}")

        train_ds = dataset_for_split(
            table=train_table,
            saprot_embeddings_dir=saprot_embeddings_dir,
            proteinmpnn_cache_dir=proteinmpnn_cache_dir,
            split="train",
            legacy_splits_csv=legacy_splits_csv,
            workbook_is_split=workbook_is_split,
        )
        val_ds = dataset_for_split(
            table=val_table,
            saprot_embeddings_dir=saprot_embeddings_dir,
            proteinmpnn_cache_dir=proteinmpnn_cache_dir,
            split="val",
            legacy_splits_csv=legacy_splits_csv,
            workbook_is_split=workbook_is_split,
        )
        test_specs = (
            []
            if args.skip_megascale_test_eval
            else resolve_test_dataset_specs(args.test_xlsx, structure_set=args.test_structure_set)
        )
        test_items: list[dict[str, Any]] = []
        for test_spec in test_specs:
            if test_spec.prediction_source_name is not None:
                continue
            test_items.append(
                {
                    "spec": test_spec,
                    "dataset": dataset_for_split(
                        table=test_spec.table,
                        saprot_embeddings_dir=saprot_embeddings_dir,
                        proteinmpnn_cache_dir=proteinmpnn_cache_dir,
                        split="test",
                        legacy_splits_csv=None,
                        workbook_is_split=True,
                        xlsx_sheet_name=test_spec.sheet_name,
                    ),
                }
            )

        if len(train_ds) == 0:
            raise RuntimeError("Training dataset is empty after matching labels, SaProt cache, and ProteinMPNN cache.")
        if len(val_ds) == 0:
            print("WARNING: Validation dataset is empty; early stopping will be disabled.")
        for test_item in test_items:
            if len(test_item["dataset"]) == 0:
                raise RuntimeError(f"Held-out test dataset is empty: {test_item['spec'].name}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=saprot_proteinmpnn_collate_fn,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=saprot_proteinmpnn_collate_fn,
            num_workers=args.num_workers,
            drop_last=False,
        )
        for test_item in test_items:
            test_item["loader"] = DataLoader(
                test_item["dataset"],
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=saprot_proteinmpnn_collate_fn,
                num_workers=args.num_workers,
                drop_last=False,
            )

        first_embedding = train_ds[0].saprot_embeddings
        if first_embedding.ndim != 2:
            raise ValueError(f"Expected SaProt embedding shape (L, D), got {tuple(first_embedding.shape)}")
        d_saprot = int(first_embedding.shape[1])
        head = make_head(d_saprot=d_saprot, args=args).to(device)
        train_criterion = composite_loss_from_args(args)
        eval_criterion = MaskedMSELoss()
        trainable_params = sum(param.numel() for param in head.parameters() if param.requires_grad)
        print(f"{head.__class__.__name__} trainable parameters: {trainable_params:,}")

        decay_params = []
        no_decay_params = []
        for name, param in head.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or param.ndim < 2:
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

        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_frac)
        scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
        effective_batch = args.batch_size * args.grad_accum
        model_config = {
            "d_saprot": d_saprot,
            "hidden_dim": args.hidden_dim,
            "aa_embed_dim": args.aa_embed_dim,
            "dropout": args.dropout,
            "attention_heads": args.attention_heads,
            "residual_blocks": args.residual_blocks,
            "local_contact_top_k": args.local_contact_top_k,
            "local_contact_cutoff": args.local_contact_cutoff,
            "local_contact_distance_scale": args.local_contact_distance_scale,
        }
        config = {
            "workflow": "saprot_proteinmpnn_intrinsic_fusion",
            "phase": 1,
            "phase_name": PHASE_NAME,
            "head_architecture": getattr(head, "architecture_name", head.__class__.__name__),
            "model_config": model_config,
            "head_params": trainable_params,
            "training_objective": train_criterion.to_config(),
            "validation_loss": "MaskedMSELoss",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size_per_device": args.batch_size,
            "grad_accumulation": args.grad_accum,
            "effective_batch_size": effective_batch,
            "warmup_frac": args.warmup_frac,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "device": str(device),
            "train_proteins": len(train_ds),
            "val_proteins": len(val_ds),
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
            "train_table": train_table,
            "val_table": val_table,
            "saprot_embeddings_dir": args.saprot_embeddings_dir,
            "proteinmpnn_cache_dir": args.proteinmpnn_cache_dir,
            "splits_csv": legacy_splits_csv,
            "dataset_route": "xlsx_per_split" if use_xlsx_splits else "csv_plus_splits",
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "dev_score_weight_megascale": args.dev_score_weight_megascale,
            "dev_score_weight_external_homolog": args.dev_score_weight_external,
            "started_at": datetime.now().isoformat(),
            "error_log_path": work_path_str(error_logger.script_log_path),
        }
        (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        best_checkpoint_score = float("-inf")
        best_checkpoint_val_global_pearson = float("-inf")
        best_checkpoint_val_dev_score = float("-inf")
        best_val_global_spearman = float("-inf")
        best_val_global_pearson = float("-inf")
        best_val_dev_score = float("-inf")
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            epoch_start = perf_counter()
            head.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            micro_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                loss = train_step(batch, head, train_criterion, device)
                (loss / args.grad_accum).backward()
                epoch_loss += float(loss.item())
                micro_steps += 1
                if micro_steps % args.grad_accum == 0 or batch_idx == len(train_loader) - 1:
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_train_loss = epoch_loss / max(micro_steps, 1)
            epoch_time = perf_counter() - epoch_start
            val_loss = float("nan")
            val_result = EvalResult()
            if len(val_ds) > 0:
                val_loss, val_result = evaluate_split(
                    val_loader,
                    head,
                    eval_criterion,
                    device,
                    dev_score_weight_megascale=args.dev_score_weight_megascale,
                    dev_score_weight_external_homolog=args.dev_score_weight_external,
                )
            val_mse, val_rmse = derive_val_mse_rmse(val_loss)
            checkpoint_score = checkpoint_selection_score(
                val_result,
                args.checkpoint_selection_metric,
            )
            epoch_log = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_mse": avg_train_loss,
                **{f"val_{key}": value for key, value in val_result.to_dict().items()},
                "checkpoint_selection_metric": args.checkpoint_selection_metric,
                "checkpoint_selection_score": checkpoint_score,
                "val_mse": val_mse,
                "val_rmse": val_rmse,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": epoch_time,
            }
            history.append(epoch_log)

            best_val_global_spearman = max(best_val_global_spearman, val_result.global_spearman)
            best_val_global_pearson = max(best_val_global_pearson, val_result.global_pearson)
            best_val_dev_score = max(best_val_dev_score, val_result.dev_score)
            improved = ""
            if (
                math.isfinite(checkpoint_score)
                and checkpoint_score > best_checkpoint_score + args.min_delta
            ):
                best_checkpoint_score = checkpoint_score
                best_checkpoint_val_global_pearson = val_result.global_pearson
                best_checkpoint_val_dev_score = val_result.dev_score
                best_epoch = epoch
                patience_counter = 0
                improved = " *"
                torch.save(head.state_dict(), ckpt_dir / "best_head.pt")
                torch.save(
                    build_fusion_package(
                        epoch=epoch,
                        head=head,
                        val_mse=val_loss,
                        val_metrics=val_result.to_full_dict(),
                        train_config=config,
                        optimizer=optimizer,
                    ),
                    ckpt_dir / "phase1_best_package.pt",
                )
                (output_dir / "best_val_results.json").write_text(
                    json.dumps(
                        {
                            "phase": 1,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_mse": val_mse,
                            "val_rmse": val_rmse,
                            "checkpoint_selection_metric": args.checkpoint_selection_metric,
                            "checkpoint_selection_score": checkpoint_score,
                            **val_result.to_full_dict(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                save_best_metrics_summary(
                    output_dir,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_result=val_result,
                    checkpoint_selection_metric=args.checkpoint_selection_metric,
                    checkpoint_selection=checkpoint_score,
                    best_checkpoint=ckpt_dir / "best_head.pt",
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
                f"lr={optimizer.param_groups[0]['lr']:.2e}  time={epoch_time:.1f}s{improved}"
            )
            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save(head.state_dict(), ckpt_dir / f"head_epoch{epoch:03d}.pt")
            if len(val_ds) > 0 and patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch}; best "
                    f"{args.checkpoint_selection_metric}={best_checkpoint_score:.4f}."
                )
                break

        if not history:
            raise RuntimeError("Training ended without recording history.")
        if best_epoch == 0:
            best_epoch = len(history)
        if not (ckpt_dir / "best_head.pt").is_file():
            torch.save(head.state_dict(), ckpt_dir / "best_head.pt")
        torch.save(head.state_dict(), ckpt_dir / "final_head.pt")
        final_metrics = {
            key.removeprefix("val_"): value
            for key, value in history[-1].items()
            if key.startswith("val_")
        }
        final_package = build_fusion_package(
            epoch=len(history),
            head=head,
            val_mse=history[-1]["val_mse"],
            val_metrics=final_metrics,
            train_config=config,
            optimizer=optimizer,
        )
        torch.save(final_package, ckpt_dir / "phase1_final_package.pt")
        if not (ckpt_dir / "phase1_best_package.pt").is_file():
            torch.save(final_package, ckpt_dir / "phase1_best_package.pt")

        if not (output_dir / "best_val_results.json").is_file():
            (output_dir / "best_val_results.json").write_text(
                json.dumps(
                    {
                        "phase": 1,
                        "epoch": best_epoch,
                        "val_loss": history[-1]["val_loss"],
                        "val_mse": history[-1]["val_mse"],
                        "val_rmse": history[-1]["val_rmse"],
                        "checkpoint_selection_metric": args.checkpoint_selection_metric,
                        "checkpoint_selection_score": history[-1]["checkpoint_selection_score"],
                        **final_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not (output_dir / "best_metrics_summary.csv").is_file():
            fallback = history[-1]
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
                        "epoch": best_epoch,
                        "val_mse": fallback["val_mse"],
                        "val_rmse": fallback["val_rmse"],
                        "val_mae": fallback.get("val_mae"),
                        "median_per_protein_spearman": fallback.get("val_median_per_protein_spearman"),
                        "median_per_protein_pearson": fallback.get("val_median_per_protein_pearson"),
                        "val_global_spearman": fallback.get("val_global_spearman"),
                        "val_global_pearson": fallback.get("val_global_pearson"),
                        "val_dev_score": fallback.get("val_dev_score"),
                        "checkpoint_selection_metric": fallback["checkpoint_selection_metric"],
                        "checkpoint_selection_score": fallback["checkpoint_selection_score"],
                        "val_megascale_validation_pearson": fallback.get("val_megascale_validation_pearson"),
                        "val_external_homolog_validation_mean_pearson": fallback.get(
                            "val_external_homolog_validation_mean_pearson"
                        ),
                        "stabilizing_ppv": fallback.get("val_stabilizing_ppv"),
                        "n_proteins": fallback.get("val_n_proteins"),
                        "n_mutations": fallback.get("val_n_mutations"),
                        "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                    }
                ],
            )

        best_head = make_head(d_saprot=d_saprot, args=model_config).to(device)
        best_head.load_state_dict(load_state_dict(ckpt_dir / "best_head.pt", map_location=device))
        phase1_test_metrics_payloads: list[dict[str, Any]] = []
        phase1_test_output_dirs: list[Path] = []
        phase1_test_metric_paths: list[Path] = []

        if len(val_ds) > 0:
            validation_rows: list[dict[str, Any]] = []
            validation_loss, validation_result = evaluate_split(
                val_loader,
                best_head,
                eval_criterion,
                device,
                mutation_prediction_rows=validation_rows,
                dev_score_weight_megascale=args.dev_score_weight_megascale,
                dev_score_weight_external_homolog=args.dev_score_weight_external,
            )
            save_megascale_test_metrics(
                output_dir=output_dir / "validation_metrics" / "phase1_best",
                phase=1,
                phase_name=PHASE_NAME,
                epoch=best_epoch,
                checkpoint_path=ckpt_dir / "best_head.pt",
                test_table=val_table,
                test_loss=validation_loss,
                test_result=validation_result,
                test_name="validation",
                test_xlsx_sheet="refined_sorted_clean",
                test_structure_set="validation",
                mutation_prediction_rows=validation_rows,
            )

        prediction_rows_by_name: dict[str, list[dict[str, Any]]] = {}
        for test_item in test_items:
            test_spec = test_item["spec"]
            mutation_prediction_rows: list[dict[str, Any]] = []
            test_loss, test_result = evaluate_split(
                test_item["loader"],
                best_head,
                eval_criterion,
                device,
                mutation_prediction_rows=mutation_prediction_rows,
                dev_score_weight_megascale=args.dev_score_weight_megascale,
                dev_score_weight_external_homolog=args.dev_score_weight_external,
            )
            test_output_dir = output_dir / "test_metrics" / test_spec.name / "phase1_best"
            payload = save_megascale_test_metrics(
                output_dir=test_output_dir,
                phase=1,
                phase_name=PHASE_NAME,
                epoch=best_epoch,
                checkpoint_path=ckpt_dir / "best_head.pt",
                test_table=test_spec.table,
                test_loss=test_loss,
                test_result=test_result,
                test_name=test_spec.name,
                test_xlsx_sheet=test_spec.sheet_name,
                test_structure_set=test_spec.structure_set,
                mutation_prediction_rows=mutation_prediction_rows,
            )
            phase1_test_metrics_payloads.append(payload)
            phase1_test_output_dirs.append(test_output_dir)
            phase1_test_metric_paths.append(test_output_dir / "metrics.json")
            prediction_rows_by_name[test_spec.name] = mutation_prediction_rows

        for test_spec in test_specs:
            if test_spec.prediction_source_name is None:
                continue
            source_rows = prediction_rows_by_name.get(test_spec.prediction_source_name)
            if source_rows is None:
                raise RuntimeError(
                    f"{test_spec.name} reuses predictions from {test_spec.prediction_source_name}, "
                    "but that source dataset was not evaluated."
                )
            test_output_dir = output_dir / "test_metrics" / test_spec.name / "phase1_best"
            payload = save_prediction_reuse_subset_metrics(
                output_dir=test_output_dir,
                phase=1,
                phase_name=PHASE_NAME,
                epoch=best_epoch,
                checkpoint_path=ckpt_dir / "best_head.pt",
                subset_spec=test_spec,
                source_rows=source_rows,
            )
            phase1_test_metrics_payloads.append(payload)
            phase1_test_output_dirs.append(test_output_dir)
            phase1_test_metric_paths.append(test_output_dir / "metrics.json")
        if phase1_test_metric_paths:
            write_megascale_test_phase_summary(
                output_dir=output_dir / "test_metrics",
                metric_paths=phase1_test_metric_paths,
            )

        write_training_history(output_dir, history)
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
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "final_val_mse": history[-1]["val_mse"],
            "final_val_rmse": history[-1]["val_rmse"],
            "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
            "final_checkpoint": work_path_str(ckpt_dir / "final_head.pt"),
            "best_phase_package": work_path_str(ckpt_dir / "phase1_best_package.pt"),
            "final_phase_package": work_path_str(ckpt_dir / "phase1_final_package.pt"),
            "test_metrics": phase1_test_metrics_payloads,
            "finished_at": datetime.now().isoformat(),
            "error_log_path": work_path_str(error_logger.script_log_path),
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        error_logger.write_run_status(
            "completed",
            summary={
                "output_dir": work_path_str(output_dir),
                "best_checkpoint": work_path_str(ckpt_dir / "best_head.pt"),
                "epochs_run": len(history),
                "best_checkpoint_score": best_checkpoint_score,
            },
        )
        print("\nTraining complete.")
        print(f"  Best checkpoint {args.checkpoint_selection_metric}: {best_checkpoint_score:.4f}")
        print(f"  Best checkpoint val dev score: {best_checkpoint_val_dev_score:.4f}")
        print(f"  Best val global Pearson overall: {best_val_global_pearson:.4f}")
        print(f"  Best checkpoint: {ckpt_dir / 'best_head.pt'}")
        print(f"  Outputs: {output_dir}")
        for payload, test_output_dir in zip(phase1_test_metrics_payloads, phase1_test_output_dirs):
            print_megascale_test_metrics_payload(payload, output_dir=test_output_dir)
    except Exception as exc:
        error_logger.exception(
            "TRAINING_FATAL",
            "train_saprot_proteinmpnn_intrinsic_fusion.py failed.",
            exc,
            context={
                "train_xlsx": args.train_xlsx,
                "val_xlsx": args.val_xlsx,
                "saprot_embeddings_dir": args.saprot_embeddings_dir,
                "proteinmpnn_cache_dir": args.proteinmpnn_cache_dir,
                "output_dir": args.output_dir,
            },
        )
        error_logger.write_run_status(
            "failed",
            summary={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "output_dir": args.output_dir,
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
