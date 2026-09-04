#!/usr/bin/env python3
"""
run_saprot_phase1_pipeline.py
-----------------------------
Single-phase SaProt workflow for the fixed validation composite raw-validation setup.

Stages:
  1. Cache SaProt structure-aware embeddings for train, validation, and held-out tests.
  2. Train the mutation-aware stability head once on cached SaProt embeddings.
  3. Evaluate the Phase 1 best package on the same held-out testing suite.

No full-forward backbone fine-tuning are run in this branch.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from models.stability_loss import add_composite_loss_args, composite_loss_cli_args  # noqa: E402
from training.megascale_test_eval import resolve_test_dataset_specs  # noqa: E402


DATASET_ROOT = "../../data/single_mutation"
DEFAULT_TRAIN_XLSX = f"{DATASET_ROOT}/final_training_set/final_training_set.xlsx"
DEFAULT_TRAIN_PDB_DIR = f"{DATASET_ROOT}/final_training_set/pdb"
DEFAULT_VAL_XLSX = (
    f"{DATASET_ROOT}/validation/"
    "megascale_plus_raw_compiled_no_non_ddg_proxy_no_ssym_inverse_removed_homologs_duplicate_filtered_validation/"
    "artifacts_excluded/"
    "megascale_plus_raw_compiled_no_non_ddg_proxy_no_ssym_inverse_removed_homologs_duplicate_filtered_validation.xlsx"
)
DEFAULT_VAL_PDB_DIR = (
    f"{DATASET_ROOT}/validation/"
    "megascale_plus_raw_compiled_no_non_ddg_proxy_no_ssym_inverse_removed_homologs_duplicate_filtered_validation/"
    "artifacts_excluded/"
    "megascale_plus_raw_compiled_no_non_ddg_proxy_no_ssym_inverse_removed_homologs_duplicate_filtered_validation_pdb"
)
DEFAULT_TEST_XLSX = "auto"
DEFAULT_TEST_PDB_DIR = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-phase SaProt cached-embedding SAAFEC workflow."
    )
    parser.add_argument("--train-xlsx", default=DEFAULT_TRAIN_XLSX)
    parser.add_argument("--train-pdb-dir", default=DEFAULT_TRAIN_PDB_DIR)
    parser.add_argument("--val-xlsx", default=DEFAULT_VAL_XLSX)
    parser.add_argument("--val-pdb-dir", default=DEFAULT_VAL_PDB_DIR)
    parser.add_argument("--test-xlsx", action="append", default=None)
    parser.add_argument("--test-pdb-dir", action="append", default=None)
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
    parser.add_argument(
        "--embeddings-dir",
        default="output/saprot_local_contact_native_parser_raw_validation_validation_composite/embeddings",
        help="SaProt embedding cache root; train/val/test share by_protein/.",
    )
    parser.add_argument(
        "--training-output-dir",
        default="output/saprot_local_contact_native_parser_raw_validation_validation_composite/phase1",
    )
    parser.add_argument("--saprot-model-dir", default="SaProt")
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--model-name", default="SaProt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
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
        choices=["global_pearson", "validation_composite"],
    )
    parser.add_argument("--dev-score-weight-megascale", type=float, default=0.5)
    parser.add_argument("--dev-score-weight-external", type=float, default=0.5)
    add_composite_loss_args(parser)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-train-cache", action="store_true")
    parser.add_argument("--skip-val-cache", action="store_true")
    parser.add_argument("--skip-test-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--skip-megascale-test-eval",
        action="store_true",
        help="Skip held-out test evaluation after Phase 1 training.",
    )
    args = parser.parse_args()
    if args.test_xlsx is None:
        args.test_xlsx = [DEFAULT_TEST_XLSX]
    if args.test_pdb_dir is None:
        args.test_pdb_dir = [DEFAULT_TEST_PDB_DIR]
    return args


def run_stage(cmd: list[str], stage_name: str) -> None:
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=WORK_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_cache_cmd(
    *,
    pdb_dir: str,
    dataset_xlsx: str,
    xlsx_sheet: str,
    embeddings_dir: str,
    saprot_model_dir: str,
    foldseek_bin: str,
    model_name: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS_ROOT / "cache" / "generate_saprot_structure_aware_cache.py"),
        "--dataset-csv",
        dataset_xlsx,
        "--xlsx-sheet",
        xlsx_sheet,
        "--pdb-dir",
        pdb_dir,
        "--output-dir",
        embeddings_dir,
        "--saprot-model-dir",
        saprot_model_dir,
        "--foldseek-bin",
        foldseek_bin,
        "--model-name",
        model_name,
        "--device",
        device,
    ]


def main() -> None:
    args = parse_args()
    embeddings_root = Path(args.embeddings_dir)
    training_output_dir = Path(args.training_output_dir)
    test_specs = (
        []
        if args.skip_megascale_test_eval and args.skip_test_cache
        else resolve_test_dataset_specs(
            args.test_xlsx,
            args.test_pdb_dir,
            structure_set=args.test_structure_set,
        )
    )

    if not args.skip_train_cache:
        run_stage(
            build_cache_cmd(
                pdb_dir=args.train_pdb_dir,
                dataset_xlsx=args.train_xlsx,
                xlsx_sheet="refined_sorted_clean",
                embeddings_dir=str(embeddings_root),
                saprot_model_dir=args.saprot_model_dir,
                foldseek_bin=args.foldseek_bin,
                model_name=args.model_name,
                device=args.device,
            ),
            "Stage 1a: Cache SaProt embeddings (train)",
        )
    else:
        print("\nSkipping Stage 1a: train SaProt cache generation")

    if not args.skip_val_cache:
        run_stage(
            build_cache_cmd(
                pdb_dir=args.val_pdb_dir,
                dataset_xlsx=args.val_xlsx,
                xlsx_sheet="refined_sorted_clean",
                embeddings_dir=str(embeddings_root),
                saprot_model_dir=args.saprot_model_dir,
                foldseek_bin=args.foldseek_bin,
                model_name=args.model_name,
                device=args.device,
            ),
            "Stage 1b: Cache SaProt embeddings (validation)",
        )
    else:
        print("\nSkipping Stage 1b: validation SaProt cache generation")

    if not args.skip_test_cache:
        for test_spec in test_specs:
            if test_spec.prediction_source_name is not None:
                print(
                    "\nSkipping Stage 1c cache generation "
                    f"({test_spec.name}; reuses {test_spec.prediction_source_name} predictions)"
                )
                continue
            if test_spec.pdb_dir is None:
                raise SystemExit(f"Missing PDB directory for test dataset {test_spec.name}.")
            run_stage(
                build_cache_cmd(
                    pdb_dir=test_spec.pdb_dir,
                    dataset_xlsx=test_spec.table,
                    xlsx_sheet=test_spec.sheet_name,
                    embeddings_dir=str(embeddings_root),
                    saprot_model_dir=args.saprot_model_dir,
                    foldseek_bin=args.foldseek_bin,
                    model_name=args.model_name,
                    device=args.device,
                ),
                f"Stage 1c: Cache SaProt embeddings ({test_spec.name})",
            )
    else:
        print("\nSkipping Stage 1c: held-out test SaProt cache generation")

    if args.skip_train:
        print("\nSkipping Stage 2: Phase 1 training")
        return

    train_cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "training" / "train_phase1_head.py"),
        "--train-xlsx",
        args.train_xlsx,
        "--val-xlsx",
        args.val_xlsx,
        "--embeddings-dir",
        str(embeddings_root / "by_protein"),
        "--output-dir",
        str(training_output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-frac",
        str(args.warmup_frac),
        "--grad-clip",
        str(args.grad_clip),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--checkpoint-selection-metric",
        args.checkpoint_selection_metric,
        "--dev-score-weight-megascale",
        str(args.dev_score_weight_megascale),
        "--dev-score-weight-external",
        str(args.dev_score_weight_external),
    ]
    train_cmd.extend(composite_loss_cli_args(args))
    train_cmd.extend(
        [
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
        ]
    )
    for test_spec in test_specs:
        train_cmd.extend(
            [
                "--test-xlsx",
                f"{test_spec.name}={test_spec.table}::{test_spec.sheet_name}",
            ]
        )
    train_cmd.extend(["--test-structure-set", args.test_structure_set])
    if args.skip_megascale_test_eval:
        train_cmd.append("--skip-megascale-test-eval")
    run_stage(train_cmd, "Stage 2: Train single-phase SaProt stability head")


if __name__ == "__main__":
    main()
