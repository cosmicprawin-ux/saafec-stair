#!/usr/bin/env python3
"""Average mutation-level predictions from multiple Phase 1 seed runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.pipeline_config import work_path_str  # noqa: E402
from core.stability_metrics import EvalResult, derive_val_mse_rmse, evaluate  # noqa: E402
from training.megascale_test_eval import SUMMARY_FIELDS  # noqa: E402


PREDICTION_FILE_PATTERNS = [
    "validation_metrics/phase1_best/mutation_predictions.csv",
    "test_metrics/*/phase1_best/mutation_predictions.csv",
]
ENSEMBLE_EXTRA_FIELDS = [
    "ensemble_n",
    "ensemble_member_predictions",
    "ensemble_member_prediction_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average mutation-level predictions from multiple seed output directories."
    )
    parser.add_argument(
        "--seed-output-dir",
        action="append",
        required=True,
        help="Phase 1 output directory for one seed. Pass once per seed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase-name",
        default="phase1_local_contact_native_parser_rawval_validation_composite_3seed_ensemble",
    )
    parser.add_argument(
        "--strict-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require identical mutation row keys in every seed for each dataset.",
    )
    return parser.parse_args()


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("protein_name", "")).strip(),
        str(row.get("model_position_1based", "")).strip(),
        str(row.get("wt_aa", "")).strip().upper(),
        str(row.get("mut_aa", "")).strip().upper(),
        str(row.get("prot_mutation_indices") or "").strip(),
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.8g}"


def discover_prediction_files(seed_output_dir: Path) -> dict[str, Path]:
    prediction_files: dict[str, Path] = {}
    for pattern in PREDICTION_FILE_PATTERNS:
        for path in seed_output_dir.glob(pattern):
            if not path.is_file():
                continue
            dataset_name = path.parent.parent.name
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                first_row = next(reader, None)
            if first_row and first_row.get("test_dataset"):
                dataset_name = str(first_row["test_dataset"])
            prediction_files[dataset_name] = path
    return prediction_files


def read_prediction_rows(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _prediction_key(row)
            if any(part == "" for part in key):
                continue
            if key in rows:
                raise ValueError(f"Duplicate prediction key {key!r} in {path}")
            rows[key] = row
    return rows


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def ensemble_rows_for_dataset(
    *,
    dataset_name: str,
    seed_prediction_paths: list[Path],
    strict_rows: bool,
    phase_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_rows = [read_prediction_rows(path) for path in seed_prediction_paths]
    key_sets = [set(rows) for rows in seed_rows]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    union_keys = set.union(*key_sets) if key_sets else set()
    if strict_rows and common_keys != union_keys:
        missing = {
            str(seed_prediction_paths[idx]): len(union_keys - keys)
            for idx, keys in enumerate(key_sets)
            if union_keys - keys
        }
        raise RuntimeError(
            f"Dataset {dataset_name} has non-identical mutation rows across seeds: {missing}"
        )

    output_rows: list[dict[str, Any]] = []
    for key in sorted(common_keys):
        base_row = dict(seed_rows[0][key])
        experimental_values = [_safe_float(rows[key].get("experimental_ddg")) for rows in seed_rows]
        first_experimental = experimental_values[0]
        if any(
            math.isfinite(value)
            and math.isfinite(first_experimental)
            and abs(value - first_experimental) > 1e-6
            for value in experimental_values[1:]
        ):
            raise RuntimeError(
                f"Dataset {dataset_name} key {key!r} has different experimental values across seeds."
            )
        predictions = [_safe_float(rows[key].get("predicted_ddg")) for rows in seed_rows]
        prediction_mean = sum(predictions) / len(predictions)
        base_row["phase_name"] = phase_name
        base_row["epoch"] = "ensemble"
        base_row["predicted_ddg"] = prediction_mean
        base_row["signed_error"] = prediction_mean - first_experimental
        base_row["absolute_error"] = abs(prediction_mean - first_experimental)
        base_row["ensemble_n"] = len(predictions)
        base_row["ensemble_member_predictions"] = ";".join(_format_float(value) for value in predictions)
        base_row["ensemble_member_prediction_std"] = _std(predictions)
        output_rows.append(base_row)

    audit = {
        "dataset": dataset_name,
        "seed_prediction_paths": [work_path_str(path) for path in seed_prediction_paths],
        "n_common_rows": len(common_keys),
        "n_union_rows": len(union_keys),
        "n_seed_rows": [len(rows) for rows in seed_rows],
        "strict_rows": strict_rows,
    }
    return output_rows, audit


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> tuple[float, EvalResult]:
    pred_by_protein: dict[str, list[float]] = {}
    true_by_protein: dict[str, list[float]] = {}
    for row in rows:
        protein = str(row.get("protein_name", "")).strip()
        if not protein:
            continue
        predicted = _safe_float(row.get("predicted_ddg"))
        experimental = _safe_float(row.get("experimental_ddg"))
        if not (math.isfinite(predicted) and math.isfinite(experimental)):
            continue
        pred_by_protein.setdefault(protein, []).append(predicted)
        true_by_protein.setdefault(protein, []).append(experimental)

    ddg_pred_list: list[torch.Tensor] = []
    ddg_true_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    protein_names: list[str] = []
    per_protein_mse: list[float] = []
    for protein in sorted(pred_by_protein):
        preds = torch.tensor(pred_by_protein[protein], dtype=torch.float32)
        trues = torch.tensor(true_by_protein[protein], dtype=torch.float32)
        if preds.numel() == 0:
            continue
        ddg_pred_list.append(preds)
        ddg_true_list.append(trues)
        mask_list.append(torch.ones_like(preds))
        protein_names.append(protein)
        per_protein_mse.append(float(((preds - trues) ** 2).mean().item()))

    result = evaluate(ddg_pred_list, ddg_true_list, mask_list, protein_names)
    loss = sum(per_protein_mse) / len(per_protein_mse) if per_protein_mse else float("nan")
    return loss, result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    for field in ENSEMBLE_EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_metrics(
    *,
    output_dir: Path,
    dataset_name: str,
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    phase_name: str,
) -> Path:
    dataset_dir = output_dir / dataset_name / "phase1_ensemble"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_rows(dataset_dir / "mutation_predictions.csv", rows)

    test_loss, result = evaluate_prediction_rows(rows)
    test_mse, test_rmse = derive_val_mse_rmse(test_loss)
    payload = {
        "phase": 1,
        "phase_name": phase_name,
        "epoch": "ensemble",
        "split": dataset_name,
        "test_dataset": dataset_name,
        "test_structure_set": rows[0].get("test_structure_set") if rows else None,
        "test_xlsx_sheet": rows[0].get("test_xlsx_sheet") if rows else None,
        "test_table": None,
        "checkpoint_path": None,
        "mutation_predictions_path": work_path_str(dataset_dir / "mutation_predictions.csv"),
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "ensemble_audit": audit,
        **result.to_full_dict(),
    }
    metrics_path = dataset_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_fields = [field for field in SUMMARY_FIELDS if field != "metrics_path"]
    summary_row = summary_row_from_payload(payload)
    with (dataset_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({field: summary_row.get(field) for field in summary_fields})

    with (dataset_dir / "per_protein_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_name", "spearman", "pearson", "rmse", "mae", "n_mutations"],
        )
        writer.writeheader()
        for detail in result.per_protein_details:
            writer.writerow(detail.to_dict())
    return metrics_path


def summary_row_from_payload(payload: dict[str, Any], metrics_path: Path | None = None) -> dict[str, Any]:
    return {
        "test_dataset": payload.get("test_dataset", payload.get("split")),
        "test_structure_set": payload.get("test_structure_set"),
        "test_xlsx_sheet": payload.get("test_xlsx_sheet"),
        "phase": payload.get("phase"),
        "phase_name": payload.get("phase_name"),
        "epoch": payload.get("epoch"),
        "test_mse": payload.get("test_mse"),
        "test_rmse": payload.get("test_rmse"),
        "test_pooled_rmse": payload.get("rmse"),
        "test_mae": payload.get("mae"),
        "test_global_spearman": payload.get("global_spearman"),
        "test_global_pearson": payload.get("global_pearson"),
        "test_median_per_protein_spearman": payload.get("median_per_protein_spearman"),
        "test_median_per_protein_pearson": payload.get("median_per_protein_pearson"),
        "test_stabilizing_ppv": payload.get("stabilizing_ppv"),
        "n_proteins": payload.get("n_proteins"),
        "n_mutations": payload.get("n_mutations"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "test_table": payload.get("test_table"),
        "metrics_path": work_path_str(metrics_path) if metrics_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    seed_output_dirs = [Path(value) for value in args.seed_output_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files_by_seed = [discover_prediction_files(path) for path in seed_output_dirs]
    dataset_names = sorted(set.intersection(*(set(files) for files in prediction_files_by_seed)))
    missing_by_seed = {
        work_path_str(seed_output_dirs[idx]): sorted(
            set.union(*(set(files) for files in prediction_files_by_seed)) - set(files)
        )
        for idx, files in enumerate(prediction_files_by_seed)
        if set.union(*(set(items) for items in prediction_files_by_seed)) - set(files)
    }
    if missing_by_seed:
        raise RuntimeError(f"Seed output directories do not contain the same datasets: {missing_by_seed}")

    metric_paths: list[Path] = []
    audits: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        seed_prediction_paths = [files[dataset_name] for files in prediction_files_by_seed]
        rows, audit = ensemble_rows_for_dataset(
            dataset_name=dataset_name,
            seed_prediction_paths=seed_prediction_paths,
            strict_rows=args.strict_rows,
            phase_name=args.phase_name,
        )
        metrics_path = write_dataset_metrics(
            output_dir=output_dir,
            dataset_name=dataset_name,
            rows=rows,
            audit=audit,
            phase_name=args.phase_name,
        )
        metric_paths.append(metrics_path)
        audits.append(audit)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for metrics_path in metric_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records.append(payload)
        rows.append(summary_row_from_payload(payload, metrics_path=metrics_path))

    records.sort(key=lambda record: str(record.get("test_dataset", "")))
    rows.sort(key=lambda row: str(row.get("test_dataset", "")))
    (output_dir / "all_ensemble_metrics.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "all_ensemble_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
    (output_dir / "ensemble_input_audit.json").write_text(
        json.dumps(audits, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote ensemble metrics for {len(metric_paths)} datasets to {output_dir}")


if __name__ == "__main__":
    main()
