#!/usr/bin/env python3
"""Run SaProt 650M PDB Tier 1 head-only supervised seeds and ensemble them."""
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
from core.pipeline_config import (  # noqa: E402
    DEFAULT_DATABASE_DIR,
    DEFAULT_FOLDSEEK_BIN,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description=(
            "Train independent Tier 1 SaProt 650M PDB supervised "
            "mutation-aware seed runs and ensemble their predictions."
        )
    )
    parser.add_argument("--train-xlsx", default=DEFAULT_TRAIN_XLSX)
    parser.add_argument("--train-pdb-dir", default=DEFAULT_TRAIN_PDB_DIR)
    parser.add_argument("--val-xlsx", default=DEFAULT_VAL_XLSX)
    parser.add_argument("--val-pdb-dir", default=DEFAULT_VAL_PDB_DIR)
    parser.add_argument("--test-xlsx", action="append", default=None)
    parser.add_argument("--test-pdb-dir", action="append", default=None)
    parser.add_argument("--test-structure-set", default="colabfold")
    parser.add_argument(
        "--output-root",
        default="output/tier1_saprot_supervised_mutation_aware_prediction",
    )
    parser.add_argument("--saprot-model-dir", default=str(DEFAULT_SAPROT_MODEL_DIR))
    parser.add_argument("--foldseek-bin", default=str(DEFAULT_FOLDSEEK_BIN))
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
    parser.add_argument("--seeds", default="1337,2027,3407")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--reuse-existing-cache",
        action="store_true",
        help="Skip cache generation for every seed and reuse output-root/embeddings.",
    )
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


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if len(seeds) < 2:
        raise SystemExit("--seeds must contain at least two comma-separated integers.")
    return seeds


def run_stage(cmd: list[str], stage_name: str) -> None:
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=WORK_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def base_pipeline_cmd(args: argparse.Namespace, *, seed: int, seed_output_dir: Path) -> list[str]:
    output_root = Path(args.output_root)
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "pipelines" / "run_saprot_phase1_pipeline.py"),
        "--train-xlsx",
        args.train_xlsx,
        "--train-pdb-dir",
        args.train_pdb_dir,
        "--val-xlsx",
        args.val_xlsx,
        "--val-pdb-dir",
        args.val_pdb_dir,
        "--test-structure-set",
        args.test_structure_set,
        "--embeddings-dir",
        str(output_root / "embeddings"),
        "--training-output-dir",
        str(seed_output_dir),
        "--saprot-model-dir",
        args.saprot_model_dir,
        "--foldseek-bin",
        args.foldseek_bin,
        "--model-name",
        args.model_name,
        "--device",
        args.device,
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
    cmd.extend(composite_loss_cli_args(args))
    cmd.extend(
        [
            "--seed",
            str(seed),
            "--num-workers",
            str(args.num_workers),
        ]
    )
    for test_xlsx in args.test_xlsx:
        cmd.extend(["--test-xlsx", test_xlsx])
    for test_pdb_dir in args.test_pdb_dir:
        cmd.extend(["--test-pdb-dir", test_pdb_dir])
    return cmd


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    output_root = Path(args.output_root)
    seed_output_dirs: list[Path] = []

    for index, seed in enumerate(seeds):
        seed_output_dir = output_root / "seeds" / f"seed_{seed}" / "phase1"
        seed_output_dirs.append(seed_output_dir)
        cmd = base_pipeline_cmd(args, seed=seed, seed_output_dir=seed_output_dir)
        if args.reuse_existing_cache or index > 0:
            cmd.extend(["--skip-train-cache", "--skip-val-cache", "--skip-test-cache"])
        run_stage(cmd, f"Train SaProt 650M PDB Tier 1 head-only seed {seed}")

    ensemble_cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "training" / "ensemble_mutation_predictions.py"),
        "--output-dir",
        str(output_root / "ensemble_metrics"),
        "--phase-name",
        "tier1_saprot_supervised_mutation_aware_prediction_ensemble",
    ]
    for seed_output_dir in seed_output_dirs:
        ensemble_cmd.extend(["--seed-output-dir", str(seed_output_dir)])
    run_stage(ensemble_cmd, "Average SaProt 650M PDB Tier 1 seed prediction tables")


if __name__ == "__main__":
    main()
