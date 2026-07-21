#!/usr/bin/env python3
"""Run the 3-seed SaProt 650M PDB + ProteinMPNN fusion model.

The runner expects SaProt embeddings and ProteinMPNN logits to have already
been cached for the input table/PDB set.
It loads each seed's checkpoint, predicts mutation-level DDG, and averages the
three seed predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.pipeline_config import ensure_output_root, resolve_output_path, work_path_str  # noqa: E402
from core.saprot_proteinmpnn_dataset import (  # noqa: E402
    SaProtProteinMPNNDataset,
    saprot_proteinmpnn_collate_fn,
)
from inference.ensemble_outputs import (  # noqa: E402
    ENSEMBLE_EXTRA_FIELDS,
    ensemble_seed_matrices,
    ensemble_seed_rows,
    write_rows,
)
from inference.model_loading import (  # noqa: E402
    choose_device,
    discover_seed_packages,
    head_from_package,
    load_package,
)
from inference.seed_inference import run_inference, write_mutation_predictions_csv  # noqa: E402
from inference.visualization_outputs import write_visualization_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", dest="input_table")
    parser.add_argument("--table-sheet", dest="table_sheet", default="refined_sorted_clean")
    parser.add_argument("--saprot-embeddings-dir", required=True)
    parser.add_argument("--proteinmpnn-cache-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="output/inference")
    parser.add_argument("--request-name", default="inference")
    parser.add_argument("--pdb-dir", default=None, help="Optional PDB directory for HTML visualizations.")
    parser.add_argument(
        "--write-visualizations",
        action="store_true",
        help="Write all-mutant matrix CSVs and standalone HTML heatmaps per protein.",
    )
    parser.add_argument(
        "--visualization-template",
        default=None,
        help="Optional base HTML file for DDG heatmaps. Defaults to scripts/visualization/ddg_heatmap.html.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not args.input_table:
        parser.error("--input-table is required")
    return args


def seed_label_from_package(package_path: Path) -> str:
    seed_label = package_path.parent.name
    if not seed_label.startswith("seed_"):
        seed_label = package_path.parent.parent.parent.name
    return seed_label


def build_dataset(args: argparse.Namespace) -> SaProtProteinMPNNDataset:
    return SaProtProteinMPNNDataset(
        mutations_table=args.input_table,
        saprot_embeddings_dir=args.saprot_embeddings_dir,
        proteinmpnn_cache_dir=args.proteinmpnn_cache_dir,
        table_sheet_name=args.table_sheet,
    )


def run_seed_packages(
    *,
    args: argparse.Namespace,
    loader: DataLoader,
    device: Any,
    output_dir: Path,
    package_paths: list[Path],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, dict[str, Any]]]]:
    seed_outputs = []
    seed_full_matrices = []
    for package_path in package_paths:
        seed_label = seed_label_from_package(package_path)
        package = load_package(package_path)
        head = head_from_package(package, device)
        rows: list[dict[str, Any]] = []
        full_matrices: dict[str, dict[str, Any]] = {}
        run_inference(
            loader,
            head,
            device,
            mutation_prediction_rows=rows,
            full_prediction_matrices=full_matrices,
        )
        context = {
            "request_name": args.request_name,
            "table_sheet": args.table_sheet,
        }
        seed_dir = output_dir / "seeds" / seed_label
        write_mutation_predictions_csv(seed_dir / "mutation_predictions.csv", rows, context=context)
        seed_outputs.append([{**context, **row} for row in rows])
        seed_full_matrices.append(full_matrices)
    return seed_outputs, seed_full_matrices


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_output_root(args.output_dir)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=saprot_proteinmpnn_collate_fn,
    )
    package_paths = discover_seed_packages(Path(args.checkpoint_dir))
    seed_outputs, seed_full_matrices = run_seed_packages(
        args=args,
        loader=loader,
        device=device,
        output_dir=output_dir,
        package_paths=package_paths,
    )

    ensemble_rows = ensemble_seed_rows(seed_outputs)
    write_rows(output_dir / "ensemble" / "mutation_predictions.csv", ensemble_rows, ENSEMBLE_EXTRA_FIELDS)
    ensemble_matrices = ensemble_seed_matrices(seed_full_matrices)
    pdb_dir = resolve_output_path(args.pdb_dir) if args.pdb_dir else None
    template_path = resolve_output_path(args.visualization_template) if args.visualization_template else None
    visualization_outputs = write_visualization_outputs(
        output_dir=output_dir,
        ensemble_matrices=ensemble_matrices,
        pdb_dir=pdb_dir,
        write_visualizations=args.write_visualizations,
        template_path=template_path,
    )
    summary = {
        "request_name": args.request_name,
        "input_table": work_path_str(args.input_table),
        "n_seed_packages": len(package_paths),
        "seed_packages": [work_path_str(path) for path in package_paths],
        "n_ensemble_rows": len(ensemble_rows),
        "n_full_matrix_proteins": len(ensemble_matrices),
        "visualization_outputs": visualization_outputs,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
