#!/usr/bin/env python3
"""Run the contact-gated calibrated-prior double-mutation workflow.

This pipeline deliberately keeps the single-mutation model fixed. It creates a
double-mutation-specific SaProt embedding cache and ProteinMPNN logits cache
from the double-mutation PDBs, exports fixed single-mutant DDG matrices from
the selected SaProt + ProteinMPNN intrinsic-fusion checkpoint(s), then trains
the contact-gated calibrated-additive-prior double-mutation residual head with
direct ProteinMPNN pair features.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.double_mutation_dataset import (  # noqa: E402
    DOUBLE_SHEET,
    load_double_mutation_workbook,
    resolve_work_path,
)
from core.pipeline_config import (  # noqa: E402
    DEFAULT_DATABASE_DIR,
    DEFAULT_FOLDSEEK_BIN,
    DEFAULT_PROTEINMPNN_CHECKPOINT,
    DEFAULT_PROTEINMPNN_SOURCE,
    DEFAULT_SAPROT_MODEL_DIR,
    DEFAULT_SINGLE_CHECKPOINT_DIR,
    DEFAULT_STRUCTURE_ROOT,
)


DEFAULT_OUTPUT_ROOT = "output/pair_corrected_double_mutation"
DEFAULT_TRAIN_XLSX = str(
    DEFAULT_DATABASE_DIR / "02_Double_Mutation/01_Training/01_MegaScale_Training_Set.xlsx"
)
DEFAULT_VAL_XLSX = str(
    DEFAULT_DATABASE_DIR / "02_Double_Mutation/02_Validation/01_MegaScale_Validation_Set.xlsx"
)
DEFAULT_TEST_XLSX = str(
    DEFAULT_DATABASE_DIR / "02_Double_Mutation/03_Testing/01_MegaScale_Test_Set.xlsx"
)
DEFAULT_TRAIN_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "double_mutation/training")
DEFAULT_VAL_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "double_mutation/validation")
DEFAULT_TEST_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "double_mutation/test")
DEFAULT_SINGLE_CHECKPOINTS = (
    str(DEFAULT_SINGLE_CHECKPOINT_DIR / "seed_1337" / "best_head.pt"),
    str(DEFAULT_SINGLE_CHECKPOINT_DIR / "seed_2027" / "best_head.pt"),
    str(DEFAULT_SINGLE_CHECKPOINT_DIR / "seed_3407" / "best_head.pt"),
)
DEFAULT_BASELINE_NAME = "saafec_stair_single_mutation_3seed_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description=(
            "Generate the required SaProt and ProteinMPNN caches, export frozen "
            "single-mutation predictions, and train the pair-corrected "
            "double-mutation extension."
        )
    )
    parser.add_argument("--train-xlsx", default=DEFAULT_TRAIN_XLSX)
    parser.add_argument("--val-xlsx", default=DEFAULT_VAL_XLSX)
    parser.add_argument("--test-xlsx", default=DEFAULT_TEST_XLSX)
    parser.add_argument("--train-pdb-dir", default=DEFAULT_TRAIN_PDB_DIR)
    parser.add_argument("--val-pdb-dir", default=DEFAULT_VAL_PDB_DIR)
    parser.add_argument("--test-pdb-dir", default=DEFAULT_TEST_PDB_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--single-checkpoint",
        dest="single_checkpoints",
        action="append",
        default=None,
        help=(
            "Single-mutation checkpoint/package to use as a baseline. Pass once per seed; "
            "comma- or semicolon-separated values are also accepted."
        ),
    )
    parser.add_argument("--baseline-name", default=DEFAULT_BASELINE_NAME)
    parser.add_argument("--saprot-model-dir", default=str(DEFAULT_SAPROT_MODEL_DIR))
    parser.add_argument("--foldseek-bin", default=str(DEFAULT_FOLDSEEK_BIN))
    parser.add_argument("--proteinmpnn-checkpoint", default=str(DEFAULT_PROTEINMPNN_CHECKPOINT))
    parser.add_argument("--proteinmpnn-source", default=str(DEFAULT_PROTEINMPNN_SOURCE))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--train-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corr-loss-weight", type=float, default=0.05)
    parser.add_argument("--correction-l2-weight", type=float, default=0.005)
    parser.add_argument("--calibration-l2-weight", type=float, default=0.001)
    parser.add_argument("--interaction-l2-weight", type=float, default=0.003)
    parser.add_argument("--far-interaction-l2-weight", type=float, default=0.012)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-proteinmpnn-cache", action="store_true")
    parser.add_argument("--skip-single-ddg", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_single_checkpoints(values: list[str] | None) -> list[str]:
    if values is None:
        return list(DEFAULT_SINGLE_CHECKPOINTS)
    checkpoints: list[str] = []
    for raw_value in values:
        for token in raw_value.replace(";", ",").split(","):
            token = token.strip()
            if token:
                checkpoints.append(token)
    if not checkpoints:
        raise ValueError("At least one --single-checkpoint path is required.")
    return checkpoints


def _pdb_lookup(pdb_dir: Path) -> dict[str, Path]:
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    output: dict[str, Path] = {}
    for path in pdb_dir.glob("*.pdb"):
        output[path.stem] = path
        output[path.stem.lower()] = path
    return output


def _copy_cache_pdb(output_path: Path, source_path: Path) -> None:
    """Copy a PDB under the logical pdb_chain stem used by the cache manifest."""
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    shutil.copy2(source_path, output_path)


def build_double_mutation_cache_manifest(
    *,
    workbooks: dict[str, Path],
    pdb_dirs: dict[str, Path],
    manifest_path: Path,
    pdb_link_dir: Path,
) -> dict[str, Any]:
    pdb_link_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_maps = {split: _pdb_lookup(path) for split, path in pdb_dirs.items()}

    proteins: dict[str, dict[str, Any]] = {}
    missing_pdbs: list[dict[str, str]] = []
    for split, workbook_path in workbooks.items():
        records = load_double_mutation_workbook(workbook_path, sheet_name=DOUBLE_SHEET)
        for protein_name, record in records.items():
            source_pdb = (
                pdb_maps[split].get(record.pdb)
                or pdb_maps[split].get(record.pdb.lower())
                or pdb_maps[split].get(protein_name)
                or pdb_maps[split].get(protein_name.lower())
            )
            if source_pdb is None:
                missing_pdbs.append(
                    {
                        "split": split,
                        "protein_name": protein_name,
                        "pdb": record.pdb,
                        "chain": record.chain,
                    }
                )
                continue

            item = proteins.setdefault(
                protein_name,
                {
                    "protein_name": protein_name,
                    "pdb": record.pdb,
                    "chain": record.chain,
                    "splits": set(),
                    "source_pdb": source_pdb,
                    "wt_sequence": record.wt_sequence,
                },
            )
            if item["wt_sequence"] != record.wt_sequence:
                raise ValueError(f"Inconsistent WT sequence for {protein_name} across splits.")
            if item["chain"] != record.chain:
                raise ValueError(f"Inconsistent chain for {protein_name} across splits.")
            item["splits"].add(split)

            cache_pdb_path = pdb_link_dir / f"{protein_name}.pdb"
            _copy_cache_pdb(cache_pdb_path, source_pdb.resolve())
            item["pdb_link"] = cache_pdb_path

    if missing_pdbs:
        preview = "; ".join(f"{row['split']}:{row['protein_name']}->{row['pdb']}" for row in missing_pdbs[:10])
        suffix = "" if len(missing_pdbs) <= 10 else f"; ... and {len(missing_pdbs) - 10} more"
        raise FileNotFoundError(f"Missing double-mutation PDB files: {preview}{suffix}")

    rows = []
    for protein_name, item in sorted(proteins.items()):
        rows.append(
            {
                "protein_name": protein_name,
                "pdb": item["pdb"],
                "chain": item["chain"],
                "splits": ";".join(sorted(item["splits"])),
                "source_pdb": str(item["source_pdb"]),
                "pdb_link": str(item["pdb_link"]),
                "wt_sequence": item["wt_sequence"],
                "wt_sequence_length": len(item["wt_sequence"]),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "protein_name",
            "pdb",
            "chain",
            "splits",
            "source_pdb",
            "pdb_link",
            "wt_sequence",
            "wt_sequence_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    split_counts = {
        split: sum(1 for row in rows if split in str(row["splits"]).split(";"))
        for split in sorted(workbooks)
    }
    return {
        "manifest_path": str(manifest_path),
        "pdb_link_dir": str(pdb_link_dir),
        "unique_proteins": len(rows),
        "split_proteins": split_counts,
    }


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_root = resolve_work_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    single_checkpoints = parse_single_checkpoints(args.single_checkpoints)
    resolved_single_checkpoints = [resolve_work_path(path) for path in single_checkpoints]

    manifest_dir = output_root / "manifests"
    manifest_path = manifest_dir / "double_mutation_embedding_manifest.csv"
    pdb_link_dir = manifest_dir / "pdb_links"
    embeddings_root = output_root / "embeddings"
    proteinmpnn_cache_root = output_root / "proteinmpnn_logits"
    single_ddg_dir = output_root / "single_ddg_matrices"
    training_output_dir = output_root / "phase1_contact_gated_calibrated_prior_residual"

    workbooks = {
        "training": resolve_work_path(args.train_xlsx),
        "validation": resolve_work_path(args.val_xlsx),
        "testing": resolve_work_path(args.test_xlsx),
    }
    pdb_dirs = {
        "training": resolve_work_path(args.train_pdb_dir),
        "validation": resolve_work_path(args.val_pdb_dir),
        "testing": resolve_work_path(args.test_pdb_dir),
    }
    manifest_summary = build_double_mutation_cache_manifest(
        workbooks=workbooks,
        pdb_dirs=pdb_dirs,
        manifest_path=manifest_path,
        pdb_link_dir=pdb_link_dir,
    )

    pipeline_summary = {
        "workflow": "double_saprot_650m_pdb_proteinmpnn_intrinsic_tier2_contact_gated_calibrated_prior_mpnn_residual_3seed_v2",
        "started_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "train_xlsx": str(workbooks["training"]),
        "val_xlsx": str(workbooks["validation"]),
        "test_xlsx": str(workbooks["testing"]),
        "single_checkpoints": [str(path) for path in resolved_single_checkpoints],
        "single_checkpoint": str(resolved_single_checkpoints[0]),
        "single_checkpoint_ensemble_n": len(resolved_single_checkpoints),
        "baseline_name": args.baseline_name,
        "embeddings_root": str(embeddings_root),
        "proteinmpnn_cache_root": str(proteinmpnn_cache_root),
        "single_ddg_dir": str(single_ddg_dir),
        "training_output_dir": str(training_output_dir),
        "manifest": manifest_summary,
        "steps": {
            "embeddings": not args.skip_embeddings,
            "proteinmpnn_cache": not args.skip_proteinmpnn_cache,
            "single_ddg": not args.skip_single_ddg,
            "train": not args.skip_train,
        },
    }
    (output_root / "double_mutation_pipeline_config.json").write_text(
        json.dumps(pipeline_summary, indent=2),
        encoding="utf-8",
    )

    if not args.skip_embeddings:
        run_command(
            [
                args.python,
                str(SCRIPTS_ROOT / "cache" / "generate_saprot_structure_aware_cache.py"),
                "--dataset-csv",
                str(manifest_path),
                "--pdb-dir",
                str(pdb_link_dir),
                "--output-dir",
                str(embeddings_root),
                "--saprot-model-dir",
                args.saprot_model_dir,
                "--foldseek-bin",
                args.foldseek_bin,
                "--device",
                args.cache_device,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_proteinmpnn_cache:
        run_command(
            [
                args.python,
                str(SCRIPTS_ROOT / "cache" / "generate_proteinmpnn_logits_cache.py"),
                "--dataset-csv",
                str(manifest_path),
                "--pdb-dir",
                str(pdb_link_dir),
                "--output-dir",
                str(proteinmpnn_cache_root),
                "--reference-cache-dir",
                str(embeddings_root / "by_protein"),
                "--proteinmpnn-checkpoint",
                args.proteinmpnn_checkpoint,
                "--proteinmpnn-source",
                args.proteinmpnn_source,
                "--device",
                args.cache_device,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_single_ddg:
        single_checkpoint_args: list[str] = []
        for checkpoint_path in resolved_single_checkpoints:
            single_checkpoint_args.extend(["--single-checkpoint", str(checkpoint_path)])
        run_command(
            [
                args.python,
                str(SCRIPTS_ROOT / "training" / "export_frozen_single_predictions.py"),
                *single_checkpoint_args,
                "--embeddings-dir",
                str(embeddings_root / "by_protein"),
                "--proteinmpnn-cache-dir",
                str(proteinmpnn_cache_root / "by_protein"),
                "--output-dir",
                str(single_ddg_dir),
                "--baseline-name",
                args.baseline_name,
                "--train-xlsx",
                str(workbooks["training"]),
                "--val-xlsx",
                str(workbooks["validation"]),
                "--test-xlsx",
                str(workbooks["testing"]),
                "--device",
                args.train_device,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_train:
        run_command(
            [
                args.python,
                str(SCRIPTS_ROOT / "training" / "train_double_mutation_saprot_local_contact.py"),
                "--train-xlsx",
                str(workbooks["training"]),
                "--val-xlsx",
                str(workbooks["validation"]),
                "--test-xlsx",
                str(workbooks["testing"]),
                "--embeddings-dir",
                str(embeddings_root / "by_protein"),
                "--single-ddg-dir",
                str(single_ddg_dir),
                "--proteinmpnn-cache-dir",
                str(proteinmpnn_cache_root / "by_protein"),
                "--output-dir",
                str(training_output_dir),
                "--baseline-name",
                args.baseline_name,
                "--device",
                args.train_device,
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--min-delta",
                str(args.min_delta),
                "--lr",
                str(args.lr),
                "--corr-loss-weight",
                str(args.corr_loss_weight),
                "--correction-l2-weight",
                str(args.correction_l2_weight),
                "--calibration-l2-weight",
                str(args.calibration_l2_weight),
                "--interaction-l2-weight",
                str(args.interaction_l2_weight),
                "--far-interaction-l2-weight",
                str(args.far_interaction_l2_weight),
                "--grad-accum",
                str(args.grad_accum),
                "--num-workers",
                str(args.num_workers),
            ],
            dry_run=args.dry_run,
        )

    print(json.dumps(pipeline_summary, indent=2))


if __name__ == "__main__":
    main()
