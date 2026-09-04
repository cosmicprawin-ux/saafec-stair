#!/usr/bin/env python3
"""Shared helpers for held-out test evaluation during training."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from core.pipeline_config import WORK_DIR, work_path_str
from core.stability_metrics import EvalResult, derive_val_mse_rmse, evaluate
from models.stability_head import AMINO_ACIDS_20


DEFAULT_TESTING_ROOT = (
    "../../data/single_mutation/testing"
)
DEFAULT_TEST_WORKBOOK_GLOB = "*_duplicate_homology_filtered.xlsx"
DEFAULT_MEGASCALE_TEST_XLSX = (
    "../../data/single_mutation/"
    "testing/megascale_test/artifacts_excluded/"
    "megascale_test_duplicate_homology_filtered.xlsx"
)
DEFAULT_FINAL_COMPILED_RAW_TEST_XLSX = (
    "../../data/single_mutation/"
    "testing/compiled_raw_compiled_databases/"
    "all_compiled_unique_no_non_ddg_proxy_no_ssym_inverse/artifacts_excluded/"
    "all_compiled_unique_no_non_ddg_proxy_no_ssym_inverse_raw_compiled_duplicate_homology_filtered_final.xlsx"
)
DEFAULT_COMPILED_RAW_COLABFOLD_PDB_DIR = (
    "../../data/single_mutation/"
    "testing/compiled_raw_compiled_databases/compiled_unique_pdbs/artifacts_excluded/"
    "compiled_raw_colabfold_best_rank001_pdb"
)
DEFAULT_TEST_XLSX = "auto"


@dataclass(frozen=True)
class TestDatasetSpec:
    name: str
    table: str
    pdb_dir: str | None = None
    sheet_name: str = "refined_sorted_clean"
    structure_set: str = "colabfold"
    prediction_source_name: str | None = None

    @property
    def reuses_prediction_rows(self) -> bool:
        return self.prediction_source_name is not None


def _slugify_test_name(name: str) -> str:
    name = re.sub(r"_duplicate_homology_filtered$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_homology_filtered$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_filtered$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return name or "test_dataset"


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORK_DIR))
    except ValueError:
        return str(path)


def _resolve_table_like_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORK_DIR / candidate).resolve()


def _compiled_unique_bundle_root(workbook: Path) -> Path | None:
    for parent in workbook.parents:
        if parent.name in {
            "compiled_unique_duplicate_homology_filtered_databases",
            "compiled_raw_compiled_databases",
        }:
            return parent
    return None


def _compiled_unique_colabfold_pdb_dir(workbook: Path) -> Path | None:
    bundle_root = _compiled_unique_bundle_root(workbook)
    if bundle_root is None:
        return None
    pdb_dir_names = [
        "compiled_unique_colabfold_best_rank001_pdb",
        "compiled_raw_colabfold_best_rank001_pdb",
    ]
    for pdb_dir_name in pdb_dir_names:
        pdb_dir = bundle_root / "compiled_unique_pdbs" / "artifacts_excluded" / pdb_dir_name
        if pdb_dir.is_dir():
            return pdb_dir
    return None


def _is_hidden_or_temporary_workbook(path: Path) -> bool:
    return path.name.startswith("~$") or any(part.startswith(".") for part in path.parts)


def _is_compiled_testing_workbook(path: Path) -> bool:
    return any(
        parent.name in {
            "compiled_unique_duplicate_homology_filtered_databases",
            "compiled_raw_compiled_databases",
        }
        for parent in path.parents
    )


def _requested_structure_sets(structure_set: str) -> list[str]:
    if structure_set == "both":
        return ["colabfold", "modelled_rank45"]
    if structure_set == "rank45_comparison":
        return ["colabfold_rank45_subset", "modelled_rank45"]
    if structure_set == "all_three":
        return ["colabfold", "colabfold_rank45_subset", "modelled_rank45"]
    return [structure_set]


def _specs_for_workbook(
    workbook: Path,
    *,
    structure_set: str,
    name: str | None = None,
    explicit_colabfold_pdb_dir: str | Path | None = None,
) -> list[TestDatasetSpec]:
    base_name = _slugify_test_name(name) if name else infer_test_dataset_name(workbook)
    specs: list[TestDatasetSpec] = []
    for requested_set in _requested_structure_sets(structure_set):
        if requested_set == "colabfold":
            pdb_dirs = sorted(workbook.parent.glob("*_colabfold_best_rank001_pdb"))
            compiled_pdb_dir = _compiled_unique_colabfold_pdb_dir(workbook)
            if explicit_colabfold_pdb_dir is not None:
                pdb_dir = _relative_or_str(_resolve_table_like_path(explicit_colabfold_pdb_dir))
            else:
                pdb_dir = (
                    _relative_or_str(pdb_dirs[0])
                    if pdb_dirs
                    else _relative_or_str(compiled_pdb_dir)
                    if compiled_pdb_dir is not None
                    else None
                )
            specs.append(
                TestDatasetSpec(
                    name=base_name,
                    table=_relative_or_str(workbook),
                    pdb_dir=pdb_dir,
                    sheet_name="refined_sorted_clean",
                    structure_set="colabfold",
                )
            )
        elif requested_set == "colabfold_rank45_subset":
            pdb_dirs = sorted(workbook.parent.glob("*_colabfold_best_rank001_pdb"))
            specs.append(
                TestDatasetSpec(
                    name=f"{base_name}__colabfold_rank45_subset",
                    table=_relative_or_str(workbook),
                    pdb_dir=_relative_or_str(pdb_dirs[0]) if pdb_dirs else None,
                    sheet_name="refined_sorted_modeled_only",
                    structure_set="colabfold_rank45_subset",
                )
            )
        elif requested_set in {"modelled_rank45", "modeled_rank45"}:
            pdb_dirs = sorted(workbook.parent.glob("*_rank4_5_modelled_aligned_pdb"))
            specs.append(
                TestDatasetSpec(
                    name=f"{base_name}__modelled_rank45",
                    table=_relative_or_str(workbook),
                    pdb_dir=_relative_or_str(pdb_dirs[0]) if pdb_dirs else None,
                    sheet_name="refined_sorted_modeled_only",
                    structure_set="modelled_rank45",
                )
            )
        else:
            raise ValueError(
                "Unsupported test structure set "
                f"{requested_set!r}; expected colabfold, colabfold_rank45_subset, "
                "modelled_rank45, both, rank45_comparison, or all_three."
            )
    return specs


def _annotate_compiled_unique_reuse(specs: list[TestDatasetSpec]) -> list[TestDatasetSpec]:
    """Mark compiled subset workbooks as evaluation-only views of all-compiled predictions."""
    by_bundle: dict[Path, list[TestDatasetSpec]] = {}
    for spec in specs:
        if spec.structure_set != "colabfold" or spec.sheet_name != "refined_sorted_clean":
            continue
        bundle_root = _compiled_unique_bundle_root(_resolve_table_like_path(spec.table))
        if bundle_root is not None:
            by_bundle.setdefault(bundle_root, []).append(spec)

    source_by_bundle: dict[Path, str] = {}
    for bundle_root, bundle_specs in by_bundle.items():
        source = next((spec for spec in bundle_specs if spec.name == "all_compiled_unique"), None)
        if source is not None:
            source_by_bundle[bundle_root] = source.name

    annotated: list[TestDatasetSpec] = []
    for spec in specs:
        bundle_root = _compiled_unique_bundle_root(_resolve_table_like_path(spec.table))
        source_name = source_by_bundle.get(bundle_root) if bundle_root is not None else None
        if (
            source_name
            and spec.name.startswith("all_compiled_unique_no_")
            and spec.name != source_name
        ):
            annotated.append(replace(spec, prediction_source_name=source_name))
        else:
            annotated.append(spec)

    return annotated


def order_test_dataset_specs(specs: list[TestDatasetSpec]) -> list[TestDatasetSpec]:
    """Keep real prediction datasets before evaluation-only subset views."""
    source_names = {spec.name for spec in specs if spec.prediction_source_name is None}
    normalized: list[TestDatasetSpec] = []
    for spec in specs:
        if spec.prediction_source_name and spec.prediction_source_name not in source_names:
            normalized.append(replace(spec, prediction_source_name=None))
        else:
            normalized.append(spec)
    return [
        *[spec for spec in normalized if spec.prediction_source_name is None],
        *[spec for spec in normalized if spec.prediction_source_name is not None],
    ]


def infer_test_dataset_name(table_path: str | Path) -> str:
    return _slugify_test_name(Path(table_path).stem)


def discover_test_dataset_specs(
    testing_root: str | Path = DEFAULT_TESTING_ROOT,
    *,
    structure_set: str = "colabfold",
) -> list[TestDatasetSpec]:
    """Discover all testing workbooks and their requested PDB folders."""
    root = Path(testing_root)
    if not root.is_absolute():
        root = WORK_DIR / root
    specs: list[TestDatasetSpec] = []
    for workbook in sorted(root.glob(f"**/artifacts_excluded/{DEFAULT_TEST_WORKBOOK_GLOB}")):
        if _is_hidden_or_temporary_workbook(workbook):
            continue
        if _is_compiled_testing_workbook(workbook):
            continue
        specs.extend(_specs_for_workbook(workbook, structure_set=structure_set))
    final_compiled_workbook = _resolve_table_like_path(DEFAULT_FINAL_COMPILED_RAW_TEST_XLSX)
    if final_compiled_workbook.is_file():
        specs.extend(
            _specs_for_workbook(
                final_compiled_workbook,
                structure_set=structure_set,
                name="all_compiled_unique_no_non_ddg_proxy_no_ssym_inverse_final",
                explicit_colabfold_pdb_dir=DEFAULT_COMPILED_RAW_COLABFOLD_PDB_DIR,
            )
        )
    return order_test_dataset_specs(_annotate_compiled_unique_reuse(specs))


def resolve_test_dataset_specs(
    raw_test_xlsx: str | list[str] | None,
    raw_test_pdb_dirs: str | list[str] | None = None,
    *,
    structure_set: str = "colabfold",
    default_sheet_name: str = "refined_sorted_clean",
) -> list[TestDatasetSpec]:
    """
    Resolve CLI test dataset inputs.

    `None`, `auto`, `all`, or `default` means discover every testing
    `*_duplicate_homology_filtered.xlsx` workbook under the current dataset's
    testing tree. Explicit paths may still point at any compatible workbook.
    Explicit values may be `path.xlsx` or `name=path.xlsx`; comma-separated
    lists are accepted for scheduler environment variables.
    """
    values: list[str] = []
    if isinstance(raw_test_xlsx, str):
        values = [raw_test_xlsx]
    elif raw_test_xlsx:
        values = list(raw_test_xlsx)
    tokens = [
        token.strip()
        for value in values
        for token in str(value).split(",")
        if token.strip()
    ]
    if not tokens or any(token.lower() in {"auto", "all", "default"} for token in tokens):
        return discover_test_dataset_specs(structure_set=structure_set)

    pdb_values: list[str] = []
    if isinstance(raw_test_pdb_dirs, str):
        pdb_values = [raw_test_pdb_dirs]
    elif raw_test_pdb_dirs:
        pdb_values = list(raw_test_pdb_dirs)
    pdb_tokens = [
        token.strip()
        for value in pdb_values
        for token in str(value).split(",")
        if token.strip() and token.strip().lower() not in {"auto", "all", "default"}
    ]

    specs: list[TestDatasetSpec] = []
    for idx, token in enumerate(tokens):
        sheet_name = default_sheet_name
        if "=" in token:
            name, table = token.split("=", 1)
            name = _slugify_test_name(name)
        else:
            table = token
            name = infer_test_dataset_name(table)
        if "::" in table:
            table, sheet_name = table.rsplit("::", 1)
            sheet_name = sheet_name.strip() or default_sheet_name
        pdb_dir = pdb_tokens[idx] if idx < len(pdb_tokens) else None
        table_path = _resolve_table_like_path(table)
        if pdb_dir is None and structure_set == "colabfold":
            compiled_pdb_dir = _compiled_unique_colabfold_pdb_dir(table_path)
            if compiled_pdb_dir is not None:
                pdb_dir = _relative_or_str(compiled_pdb_dir)
        specs.append(
            TestDatasetSpec(
                name=name,
                table=table,
                pdb_dir=pdb_dir,
                sheet_name=sheet_name,
                structure_set=structure_set,
            )
        )
    return order_test_dataset_specs(_annotate_compiled_unique_reuse(specs))


SUMMARY_FIELDS = [
    "test_dataset",
    "test_structure_set",
    "test_xlsx_sheet",
    "phase",
    "phase_name",
    "epoch",
    "test_mse",
    "test_rmse",
    "test_pooled_rmse",
    "test_mae",
    "test_global_spearman",
    "test_global_pearson",
    "test_median_per_protein_spearman",
    "test_median_per_protein_pearson",
    "test_stabilizing_ppv",
    "n_proteins",
    "n_mutations",
    "checkpoint_path",
    "test_table",
    "metrics_path",
]


MUTATION_PREDICTION_FIELDS = [
    "test_dataset",
    "test_structure_set",
    "test_xlsx_sheet",
    "phase",
    "phase_name",
    "epoch",
    "protein_name",
    "model_position_1based",
    "wt_aa",
    "mut_aa",
    "mutation",
    "experimental_ddg",
    "predicted_ddg",
    "signed_error",
    "absolute_error",
    "duplicate_count",
    "source_ddg_values",
    "source_raw_positions_1based",
    "source_position_sources",
    "pdb_residue_ids",
    "pdb_resseqs",
    "pdb_ins_codes",
    "resolution_methods",
    "prot_mutation_indices",
    "structure_ids",
    "chains",
]


def _join_unique(values: list[Any]) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in {None, ""}:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            output.append(text)
    return ";".join(output)


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return ""


def _resolution_records_by_mutation(
    mutation_resolution: dict[str, Any] | None,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    if not mutation_resolution:
        return grouped
    for record in mutation_resolution.get("records", []):
        if not record.get("resolved"):
            continue
        model_index = record.get("model_index")
        mut_aa = record.get("mut_aa")
        if model_index is None or not mut_aa:
            continue
        try:
            key = (int(model_index), str(mut_aa))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(record)
    return grouped


def build_mutation_prediction_rows_for_protein(
    *,
    ddg_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    wt_sequence: str,
    protein_name: str,
    mutation_resolution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build source-row mutation predictions from the exact masked entries used in evaluation."""
    ddg_pred = ddg_pred.detach().cpu()
    target = target.detach().cpu()
    mask = mask.detach().cpu()
    records_by_key = _resolution_records_by_mutation(mutation_resolution)

    rows: list[dict[str, Any]] = []
    masked_positions = (mask > 0).nonzero(as_tuple=False)
    for pos_idx, mut_idx in masked_positions.tolist():
        wt_aa = wt_sequence[pos_idx] if pos_idx < len(wt_sequence) else ""
        mut_aa = AMINO_ACIDS_20[mut_idx]
        experimental = float(target[pos_idx, mut_idx].item())
        predicted = float(ddg_pred[pos_idx, mut_idx].item())
        source_records = records_by_key.get((pos_idx + 1, mut_aa), [])
        if source_records:
            for record in source_records:
                try:
                    source_experimental = float(record.get("ddg"))
                except (TypeError, ValueError):
                    source_experimental = experimental
                rows.append(
                    {
                        "protein_name": protein_name,
                        "model_position_1based": pos_idx + 1,
                        "wt_aa": wt_aa,
                        "mut_aa": mut_aa,
                        "mutation": f"{wt_aa}{pos_idx + 1}{mut_aa}",
                        "experimental_ddg": source_experimental,
                        "predicted_ddg": predicted,
                        "signed_error": predicted - source_experimental,
                        "absolute_error": abs(predicted - source_experimental),
                        "duplicate_count": 1,
                        "source_ddg_values": _format_float(record.get("ddg")),
                        "source_raw_positions_1based": _join_unique(
                            [record.get("raw_position_1based")]
                        ),
                        "source_position_sources": _join_unique(
                            [record.get("position_source")]
                        ),
                        "pdb_residue_ids": _join_unique([record.get("pdb_residue_id")]),
                        "pdb_resseqs": _join_unique([record.get("pdb_resseq")]),
                        "pdb_ins_codes": _join_unique([record.get("pdb_ins_code")]),
                        "resolution_methods": _join_unique(
                            [record.get("resolution_method")]
                        ),
                        "prot_mutation_indices": _join_unique(
                            [record.get("prot_mutation_index")]
                        ),
                        "structure_ids": _join_unique([record.get("structure_id")]),
                        "chains": _join_unique([record.get("chain")]),
                    }
                )
            continue
        rows.append(
            {
                "protein_name": protein_name,
                "model_position_1based": pos_idx + 1,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "mutation": f"{wt_aa}{pos_idx + 1}{mut_aa}",
                "experimental_ddg": experimental,
                "predicted_ddg": predicted,
                "signed_error": predicted - experimental,
                "absolute_error": abs(predicted - experimental),
                "duplicate_count": 1,
                "source_ddg_values": _format_float(experimental),
                "source_raw_positions_1based": _join_unique(
                    [record.get("raw_position_1based") for record in source_records]
                ),
                "source_position_sources": _join_unique(
                    [record.get("position_source") for record in source_records]
                ),
                "pdb_residue_ids": _join_unique(
                    [record.get("pdb_residue_id") for record in source_records]
                ),
                "pdb_resseqs": _join_unique(
                    [record.get("pdb_resseq") for record in source_records]
                ),
                "pdb_ins_codes": _join_unique(
                    [record.get("pdb_ins_code") for record in source_records]
                ),
                "resolution_methods": _join_unique(
                    [record.get("resolution_method") for record in source_records]
                ),
                "prot_mutation_indices": _join_unique(
                    [record.get("prot_mutation_index") for record in source_records]
                ),
                "structure_ids": _join_unique(
                    [record.get("structure_id") for record in source_records]
                ),
                "chains": _join_unique([record.get("chain") for record in source_records]),
            }
        )
    return rows


def write_mutation_predictions_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
) -> None:
    """Write the row-level prediction/export table for one evaluated test dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = context or {}
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MUTATION_PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            combined = {**context, **row}
            writer.writerow({field: combined.get(field) for field in MUTATION_PREDICTION_FIELDS})


def _split_semicolon_values(value: Any) -> list[str]:
    if value in {None, ""}:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _format_position_key(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _prediction_row_keys(row: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    protein = str(row.get("protein_name", "")).strip()
    wt_aa = str(row.get("wt_aa", "")).strip().upper()
    mut_aa = str(row.get("mut_aa", "")).strip().upper()
    if not protein or not wt_aa or not mut_aa:
        return set()

    raw_positions = (
        _split_semicolon_values(row.get("source_raw_positions_1based"))
        or _split_semicolon_values(row.get("pdb_resseqs"))
        or _split_semicolon_values(row.get("model_position_1based"))
    )
    keys: set[tuple[str, str, str, str]] = set()
    for position in raw_positions:
        pos_key = _format_position_key(position)
        if pos_key is not None:
            keys.add((protein, pos_key, wt_aa, mut_aa))
    return keys


def _subset_workbook_keys(spec: TestDatasetSpec) -> set[tuple[str, str, str, str]]:
    from core.megascale_dataset import load_workbook_records_xlsx  # noqa: WPS433

    keys: set[tuple[str, str, str, str]] = set()
    for protein_name, record in load_workbook_records_xlsx(
        spec.table,
        sheet_name=spec.sheet_name,
    ).items():
        for mutation in record.mutations:
            pos_key = _format_position_key(
                mutation.get("position_raw_1based")
                or mutation.get("pdb_resseq")
                or mutation.get("position", 0) + 1
            )
            wt_aa = str(mutation.get("wt_aa", "")).strip().upper()
            mut_aa = str(mutation.get("mut_aa", "")).strip().upper()
            if pos_key and wt_aa and mut_aa:
                keys.add((protein_name, pos_key, wt_aa, mut_aa))
    return keys


def evaluate_prediction_rows_subset(
    *,
    source_rows: list[dict[str, Any]],
    subset_spec: TestDatasetSpec,
) -> tuple[float, EvalResult, list[dict[str, Any]]]:
    """
    Compute metrics for a subset workbook from already-generated mutation predictions.

    This is used for the compiled unique databases where
    `all_compiled_unique_duplicate_homology_filtered.xlsx` is evaluated once, and
    the no-proxy / no-proxy-no-SSYM-inverse workbooks are strict subsets of those
    mutation rows.
    """
    subset_keys = _subset_workbook_keys(subset_spec)
    if not subset_keys:
        raise RuntimeError(
            f"Subset workbook {subset_spec.name} has no mutation keys in "
            f"{subset_spec.table}::{subset_spec.sheet_name}."
        )

    filtered_rows: list[dict[str, Any]] = []
    seen_row_ids: set[int] = set()
    for row in source_rows:
        if id(row) in seen_row_ids:
            continue
        if _prediction_row_keys(row) & subset_keys:
            filtered_rows.append(row)
            seen_row_ids.add(id(row))

    if not filtered_rows:
        raise RuntimeError(
            f"No prediction rows from {subset_spec.prediction_source_name} matched "
            f"subset workbook {subset_spec.name}."
        )

    pred_by_protein: dict[str, list[float]] = {}
    true_by_protein: dict[str, list[float]] = {}
    for row in filtered_rows:
        protein = str(row.get("protein_name", "")).strip()
        if not protein:
            continue
        try:
            pred = float(row["predicted_ddg"])
            true = float(row["experimental_ddg"])
        except (KeyError, TypeError, ValueError):
            continue
        pred_by_protein.setdefault(protein, []).append(pred)
        true_by_protein.setdefault(protein, []).append(true)

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
    test_loss = (
        sum(per_protein_mse) / len(per_protein_mse)
        if per_protein_mse
        else float("nan")
    )
    return test_loss, result, filtered_rows


def save_prediction_reuse_subset_metrics(
    *,
    output_dir: Path,
    phase: int,
    phase_name: str,
    epoch: int,
    checkpoint_path: Path,
    subset_spec: TestDatasetSpec,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    test_loss, test_result, filtered_rows = evaluate_prediction_rows_subset(
        source_rows=source_rows,
        subset_spec=subset_spec,
    )
    payload = save_megascale_test_metrics(
        output_dir=output_dir,
        phase=phase,
        phase_name=phase_name,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        test_table=subset_spec.table,
        test_loss=test_loss,
        test_result=test_result,
        test_name=subset_spec.name,
        test_xlsx_sheet=subset_spec.sheet_name,
        test_structure_set=subset_spec.structure_set,
        mutation_prediction_rows=filtered_rows,
    )
    payload["prediction_reuse_source_dataset"] = subset_spec.prediction_source_name
    payload["prediction_reuse_mode"] = "filtered_mutation_predictions"
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def summary_row_from_payload(
    payload: dict[str, Any],
    *,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Return one CSV-friendly row from a saved MegaScale test metrics payload."""
    return {
        "phase": payload.get("phase"),
        "test_dataset": payload.get("test_dataset", payload.get("split")),
        "test_structure_set": payload.get("test_structure_set"),
        "test_xlsx_sheet": payload.get("test_xlsx_sheet"),
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


def write_megascale_test_phase_summary(
    *,
    output_dir: Path,
    metric_paths: list[Path],
    summary_stem: str = "all_phase_test_metrics",
) -> list[dict[str, Any]]:
    """Write a consolidated one-row-per-phase test summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in metric_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(payload)
        rows.append(summary_row_from_payload(payload, metrics_path=path))

    records.sort(key=lambda record: (record.get("test_dataset", ""), int(record.get("phase", 10**9))))
    rows.sort(key=lambda row: (row.get("test_dataset", ""), int(row.get("phase") or 10**9)))

    (output_dir / f"{summary_stem}.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    with open(
        output_dir / f"{summary_stem}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})

    return records


def save_megascale_test_metrics(
    *,
    output_dir: Path,
    phase: int,
    phase_name: str,
    epoch: int,
    checkpoint_path: Path,
    test_table: str,
    test_loss: float,
    test_result: EvalResult,
    test_name: str = "megascale_test",
    test_xlsx_sheet: str = "refined_sorted_clean",
    test_structure_set: str = "colabfold",
    mutation_prediction_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save JSON, summary CSV, per-protein CSV, and optional mutation-level CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_mse, test_rmse = derive_val_mse_rmse(test_loss)
    mutation_predictions_path = output_dir / "mutation_predictions.csv"
    payload: dict[str, Any] = {
        "phase": phase,
        "phase_name": phase_name,
        "epoch": epoch,
        "split": test_name,
        "test_dataset": test_name,
        "test_structure_set": test_structure_set,
        "test_xlsx_sheet": test_xlsx_sheet,
        "test_table": test_table,
        "checkpoint_path": work_path_str(checkpoint_path),
        "mutation_predictions_path": (
            work_path_str(mutation_predictions_path)
            if mutation_prediction_rows is not None
            else None
        ),
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        **test_result.to_full_dict(),
    }

    if mutation_prediction_rows is not None:
        write_mutation_predictions_csv(
            mutation_predictions_path,
            mutation_prediction_rows,
            context={
                "test_dataset": test_name,
                "test_structure_set": test_structure_set,
                "test_xlsx_sheet": test_xlsx_sheet,
                "phase": phase,
                "phase_name": phase_name,
                "epoch": epoch,
            },
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    summary_fields = [field for field in SUMMARY_FIELDS if field != "metrics_path"]
    summary_row = summary_row_from_payload(payload)
    with open(output_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({field: summary_row.get(field) for field in summary_fields})

    per_protein_fields = ["protein_name", "spearman", "pearson", "rmse", "mae", "n_mutations"]
    with open(output_dir / "per_protein_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_protein_fields)
        writer.writeheader()
        for detail in test_result.per_protein_details:
            writer.writerow(detail.to_dict())

    return payload


def print_megascale_test_metrics(
    *,
    phase: int,
    phase_name: str,
    epoch: int,
    test_result: EvalResult,
    test_loss: float,
    output_dir: Path,
    test_name: str = "megascale_test",
) -> None:
    """Print the held-out test metrics so they appear in scheduler .out files."""
    test_mse, test_rmse = derive_val_mse_rmse(test_loss)
    print(f"\nHeld-out test evaluation: {test_name}")
    print(f"  phase={phase}  phase_name={phase_name}  best_epoch={epoch}")
    print(
        f"  test_global_spearman={test_result.global_spearman:.4f}  "
        f"test_global_pearson={test_result.global_pearson:.4f}  "
        f"test_median_per_protein_spearman={test_result.median_per_protein_spearman:.4f}  "
        f"test_median_per_protein_pearson={test_result.median_per_protein_pearson:.4f}"
    )
    print(
        f"  test_mse={test_mse:.5f}  test_rmse={test_rmse:.5f}  "
        f"test_pooled_rmse={test_result.rmse:.5f}  test_mae={test_result.mae:.5f}  "
        f"test_stabilizing_ppv={test_result.stabilizing_ppv:.4f}"
    )
    print(
        f"  test_proteins={test_result.n_proteins}  "
        f"test_mutations={test_result.n_mutations}  "
        f"test_metrics_dir={output_dir}"
    )


def print_megascale_test_metrics_payload(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    """Print a saved held-out test metric payload in the standard .out format."""
    test_name = payload.get("test_dataset", payload.get("split", "test"))
    structure_set = payload.get("test_structure_set")
    xlsx_sheet = payload.get("test_xlsx_sheet")
    print(
        f"\nHeld-out test evaluation: {test_name}"
    )
    if structure_set or xlsx_sheet:
        print(
            f"  structure_set={structure_set or 'unknown'}  "
            f"xlsx_sheet={xlsx_sheet or 'unknown'}"
        )
    print(
        f"  phase={payload['phase']}  phase_name={payload['phase_name']}  "
        f"best_epoch={payload['epoch']}"
    )
    print(
        f"  test_global_spearman={payload['global_spearman']:.4f}  "
        f"test_global_pearson={payload['global_pearson']:.4f}  "
        f"test_median_per_protein_spearman={payload['median_per_protein_spearman']:.4f}  "
        f"test_median_per_protein_pearson={payload['median_per_protein_pearson']:.4f}"
    )
    print(
        f"  test_mse={payload['test_mse']:.5f}  "
        f"test_rmse={payload['test_rmse']:.5f}  "
        f"test_pooled_rmse={payload['rmse']:.5f}  "
        f"test_mae={payload['mae']:.5f}  "
        f"test_stabilizing_ppv={payload['stabilizing_ppv']:.4f}"
    )
    print(
        f"  test_proteins={payload['n_proteins']}  "
        f"test_mutations={payload['n_mutations']}  "
        f"test_metrics_dir={output_dir}"
    )
