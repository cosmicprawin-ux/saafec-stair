#!/usr/bin/env python3
"""Run one SaProt + ProteinMPNN intrinsic-fusion training/evaluation job."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from models.stability_loss import composite_loss_cli_args  # noqa: E402
from training.megascale_test_eval import resolve_test_dataset_specs  # noqa: E402
from core.pipeline_config import (  # noqa: E402
    DEFAULT_DATABASE_DIR,
    DEFAULT_FOLDSEEK_BIN,
    DEFAULT_PROTEINMPNN_CHECKPOINT,
    DEFAULT_PROTEINMPNN_SOURCE,
    DEFAULT_SAPROT_MODEL_DIR,
    DEFAULT_STRUCTURE_ROOT,
)


DEFAULT_TRAIN_XLSX = str(
    DEFAULT_DATABASE_DIR / "01_Single_Mutation/01_Training/01_MegaScale_Training_Set.xlsx"
)
DEFAULT_VAL_XLSX = str(
    DEFAULT_DATABASE_DIR / "01_Single_Mutation/02_Validation/03_Final_Validation_Set.xlsx"
)
DEFAULT_TEST_XLSX = str(
    DEFAULT_DATABASE_DIR / "01_Single_Mutation/03_Testing/04_Final_Test_Set.xlsx"
)
DEFAULT_TRAIN_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "single_mutation/training")
DEFAULT_VAL_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "single_mutation/validation")
DEFAULT_TEST_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "single_mutation/test")
DEFAULT_OUTPUT_ROOT = "output/tier2_saprot_proteinmpnn_augmented_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single SaProt + ProteinMPNN intrinsic-fusion SAAFEC job."
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
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--saprot-embeddings-dir",
        default=None,
        help="Defaults to output-root/embeddings.",
    )
    parser.add_argument(
        "--proteinmpnn-cache-dir",
        default=None,
        help="Defaults to output-root/proteinmpnn_logits.",
    )
    parser.add_argument(
        "--training-output-dir",
        default=None,
        help="Defaults to output-root/phase1.",
    )
    parser.add_argument("--saprot-model-dir", default=str(DEFAULT_SAPROT_MODEL_DIR))
    parser.add_argument("--foldseek-bin", default=str(DEFAULT_FOLDSEEK_BIN))
    parser.add_argument("--model-name", default="SaProt")
    parser.add_argument("--proteinmpnn-checkpoint", default=str(DEFAULT_PROTEINMPNN_CHECKPOINT))
    parser.add_argument("--proteinmpnn-source", default=str(DEFAULT_PROTEINMPNN_SOURCE))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--aa-embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--local-contact-top-k", type=int, default=16)
    parser.add_argument("--local-contact-cutoff", type=float, default=10.0)
    parser.add_argument("--local-contact-distance-scale", type=float, default=4.0)

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

    # Keep training-objective knobs identical to the existing mutation-aware head.
    from models.stability_loss import add_composite_loss_args  # noqa: WPS433

    add_composite_loss_args(parser)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-train-cache", action="store_true")
    parser.add_argument("--skip-val-cache", action="store_true")
    parser.add_argument("--skip-test-cache", action="store_true")
    parser.add_argument("--skip-saprot-cache", action="store_true")
    parser.add_argument("--skip-proteinmpnn-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-megascale-test-eval", action="store_true")
    args = parser.parse_args()
    if args.test_xlsx is None:
        args.test_xlsx = [DEFAULT_TEST_XLSX]
    if args.test_pdb_dir is None:
        args.test_pdb_dir = [DEFAULT_TEST_PDB_DIR]
    if args.dev_score_weight_megascale < 0 or args.dev_score_weight_external < 0:
        parser.error("validation-score weights must be non-negative.")
    if args.dev_score_weight_megascale + args.dev_score_weight_external <= 0:
        parser.error("At least one validation-score weight must be positive.")
    return args


def run_stage(cmd: list[str], stage_name: str) -> None:
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=WORK_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_saprot_cache_cmd(
    *,
    pdb_dir: str,
    dataset_xlsx: str,
    xlsx_sheet: str,
    output_dir: Path,
    args: argparse.Namespace,
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
        str(output_dir),
        "--saprot-model-dir",
        args.saprot_model_dir,
        "--foldseek-bin",
        args.foldseek_bin,
        "--model-name",
        args.model_name,
        "--device",
        args.device,
    ]


def build_proteinmpnn_cache_cmd(
    *,
    pdb_dir: str,
    dataset_xlsx: str,
    xlsx_sheet: str,
    output_dir: Path,
    reference_cache_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS_ROOT / "cache" / "generate_proteinmpnn_logits_cache.py"),
        "--dataset-csv",
        dataset_xlsx,
        "--xlsx-sheet",
        xlsx_sheet,
        "--pdb-dir",
        pdb_dir,
        "--output-dir",
        str(output_dir),
        "--reference-cache-dir",
        str(reference_cache_dir),
        "--proteinmpnn-checkpoint",
        args.proteinmpnn_checkpoint,
        "--proteinmpnn-source",
        args.proteinmpnn_source,
        "--device",
        args.device,
    ]


def maybe_cache_split(
    *,
    label: str,
    skip: bool,
    pdb_dir: str | None,
    dataset_xlsx: str,
    xlsx_sheet: str,
    saprot_embeddings_root: Path,
    proteinmpnn_cache_root: Path,
    args: argparse.Namespace,
) -> None:
    if skip:
        print(f"\nSkipping cache generation ({label})")
        return
    if pdb_dir is None:
        raise SystemExit(f"Missing PDB directory for {label}.")
    if not args.skip_saprot_cache:
        run_stage(
            build_saprot_cache_cmd(
                pdb_dir=pdb_dir,
                dataset_xlsx=dataset_xlsx,
                xlsx_sheet=xlsx_sheet,
                output_dir=saprot_embeddings_root,
                args=args,
            ),
            f"Cache SaProt embeddings ({label})",
        )
    else:
        print(f"\nSkipping SaProt cache ({label})")
    if not args.skip_proteinmpnn_cache:
        run_stage(
            build_proteinmpnn_cache_cmd(
                pdb_dir=pdb_dir,
                dataset_xlsx=dataset_xlsx,
                xlsx_sheet=xlsx_sheet,
                output_dir=proteinmpnn_cache_root,
                reference_cache_dir=saprot_embeddings_root / "by_protein",
                args=args,
            ),
            f"Cache ProteinMPNN logits ({label})",
        )
    else:
        print(f"\nSkipping ProteinMPNN cache ({label})")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    saprot_embeddings_root = (
        Path(args.saprot_embeddings_dir)
        if args.saprot_embeddings_dir is not None
        else output_root / "embeddings"
    )
    proteinmpnn_cache_root = (
        Path(args.proteinmpnn_cache_dir)
        if args.proteinmpnn_cache_dir is not None
        else output_root / "proteinmpnn_logits"
    )
    training_output_dir = (
        Path(args.training_output_dir)
        if args.training_output_dir is not None
        else output_root / "phase1"
    )
    test_specs = (
        []
        if args.skip_megascale_test_eval and args.skip_test_cache
        else resolve_test_dataset_specs(
            args.test_xlsx,
            args.test_pdb_dir,
            structure_set=args.test_structure_set,
        )
    )

    maybe_cache_split(
        label="train",
        skip=args.skip_train_cache,
        pdb_dir=args.train_pdb_dir,
        dataset_xlsx=args.train_xlsx,
        xlsx_sheet="refined_sorted_clean",
        saprot_embeddings_root=saprot_embeddings_root,
        proteinmpnn_cache_root=proteinmpnn_cache_root,
        args=args,
    )
    maybe_cache_split(
        label="validation",
        skip=args.skip_val_cache,
        pdb_dir=args.val_pdb_dir,
        dataset_xlsx=args.val_xlsx,
        xlsx_sheet="refined_sorted_clean",
        saprot_embeddings_root=saprot_embeddings_root,
        proteinmpnn_cache_root=proteinmpnn_cache_root,
        args=args,
    )
    if not args.skip_test_cache:
        for test_spec in test_specs:
            if test_spec.prediction_source_name is not None:
                print(
                    "\nSkipping test cache generation "
                    f"({test_spec.name}; reuses {test_spec.prediction_source_name} predictions)"
                )
                continue
            maybe_cache_split(
                label=test_spec.name,
                skip=False,
                pdb_dir=test_spec.pdb_dir,
                dataset_xlsx=test_spec.table,
                xlsx_sheet=test_spec.sheet_name,
                saprot_embeddings_root=saprot_embeddings_root,
                proteinmpnn_cache_root=proteinmpnn_cache_root,
                args=args,
            )
    else:
        print("\nSkipping held-out test cache generation")

    if args.skip_train:
        print("\nSkipping fusion training")
        return

    train_cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "training" / "train_saprot_proteinmpnn_intrinsic_fusion.py"),
        "--train-xlsx",
        args.train_xlsx,
        "--val-xlsx",
        args.val_xlsx,
        "--saprot-embeddings-dir",
        str(saprot_embeddings_root / "by_protein"),
        "--proteinmpnn-cache-dir",
        str(proteinmpnn_cache_root / "by_protein"),
        "--output-dir",
        str(training_output_dir),
        "--hidden-dim",
        str(args.hidden_dim),
        "--aa-embed-dim",
        str(args.aa_embed_dim),
        "--dropout",
        str(args.dropout),
        "--attention-heads",
        str(args.attention_heads),
        "--residual-blocks",
        str(args.residual_blocks),
        "--local-contact-top-k",
        str(args.local_contact_top_k),
        "--local-contact-cutoff",
        str(args.local_contact_cutoff),
        "--local-contact-distance-scale",
        str(args.local_contact_distance_scale),
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
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    train_cmd.extend(composite_loss_cli_args(args))
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
    run_stage(train_cmd, "Train SaProt + ProteinMPNN intrinsic fusion head")


if __name__ == "__main__":
    main()
