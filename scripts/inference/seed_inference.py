"""Single-seed model inference and mutation-level CSV output."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.amino_acids import AMINO_ACIDS_20


MUTATION_PREDICTION_FIELDS = [
    "request_name",
    "table_sheet",
    "pdb",
    "chain",
    "model_position_1based",
    "wt_aa",
    "mut_aa",
    "mutation",
    "predicted_ddg",
    "duplicate_count",
    "source_raw_positions_1based",
    "source_position_sources",
    "pdb_residue_ids",
    "pdb_resseqs",
    "pdb_ins_codes",
    "resolution_methods",
    "prot_mutation_indices",
    "structure_ids",
]
INTERNAL_OUTPUT_FIELDS: set[str] = set()
PREDICTION_ONLY_OMIT_FIELDS = [
    "duplicate_count",
    "source_raw_positions_1based",
    "source_position_sources",
    "resolution_methods",
    "prot_mutation_indices",
    "structure_ids",
]
CSV_DISPLAY_FIELD_NAMES = {
    "predicted_ddg": "predicted_ΔΔG_kcal_per_mol",
}


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


def mutation_prediction_fieldnames(
    rows: list[dict[str, Any]],
    extra_fields: list[str] | None = None,
) -> list[str]:
    omitted = set(PREDICTION_ONLY_OMIT_FIELDS)
    omitted.update(INTERNAL_OUTPUT_FIELDS)
    fieldnames = [
        field
        for field in MUTATION_PREDICTION_FIELDS
        if field not in omitted
    ]
    for field in extra_fields or []:
        if field not in fieldnames and field not in omitted:
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in fieldnames and field not in omitted:
                fieldnames.append(field)
    return fieldnames


def csv_display_field_name(field: str) -> str:
    return CSV_DISPLAY_FIELD_NAMES.get(field, field)


def write_display_rows(handle: Any, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    display_fieldnames = [csv_display_field_name(field) for field in fieldnames]
    writer = csv.DictWriter(handle, fieldnames=display_fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                csv_display_field_name(field): row.get(field)
                for field in fieldnames
            }
        )


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
    mask: torch.Tensor,
    wt_sequence: str,
    protein_name: str,
    mutation_resolution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ddg_pred = ddg_pred.detach().cpu()
    mask = mask.detach().cpu()
    records_by_key = _resolution_records_by_mutation(mutation_resolution)

    rows: list[dict[str, Any]] = []
    masked_positions = (mask > 0).nonzero(as_tuple=False)
    for pos_idx, mut_idx in masked_positions.tolist():
        wt_aa = wt_sequence[pos_idx] if pos_idx < len(wt_sequence) else ""
        mut_aa = AMINO_ACIDS_20[mut_idx]
        predicted = float(ddg_pred[pos_idx, mut_idx].item())
        source_records = records_by_key.get((pos_idx + 1, mut_aa), [])
        row = {
            "pdb": next((str(record.get("structure_id")) for record in source_records if record.get("structure_id")), protein_name),
            "chain": next((str(record.get("chain")) for record in source_records if record.get("chain")), ""),
            "model_position_1based": pos_idx + 1,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "mutation": f"{wt_aa}{pos_idx + 1}{mut_aa}",
            "predicted_ddg": predicted,
            "duplicate_count": len(source_records) if source_records else 1,
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
        }
        rows.append(row)
    return rows


def write_mutation_predictions_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = context or {}
    combined_rows = [{**context, **row} for row in rows]
    fieldnames = mutation_prediction_fieldnames(combined_rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        write_display_rows(handle, fieldnames, combined_rows)


def predict_batch(
    batch: dict[str, Any],
    head: torch.nn.Module,
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


@torch.no_grad()
def run_inference(
    loader: DataLoader,
    head: torch.nn.Module,
    device: torch.device,
    *,
    mutation_prediction_rows: list[dict[str, Any]] | None = None,
    full_prediction_matrices: dict[str, dict[str, Any]] | None = None,
) -> None:
    head.eval()

    for batch in loader:
        mutation_masks = batch["mutation_masks"].to(device)
        names = batch["names"]
        sequences = batch["sequences"]
        mutation_resolutions = batch.get("mutation_resolutions", [None] * len(names))
        ddg_pred = predict_batch(batch, head, device)

        for i in range(ddg_pred.shape[0]):
            length = int(batch["lengths"][i])
            if full_prediction_matrices is not None:
                full_prediction_matrices[names[i]] = {
                    "protein_name": names[i],
                    "sequence": sequences[i],
                    "predicted_ddg_matrix": ddg_pred[i, :length].detach().cpu(),
                }
            if mutation_prediction_rows is not None:
                mutation_prediction_rows.extend(
                    build_mutation_prediction_rows_for_protein(
                        ddg_pred=ddg_pred[i, :length],
                        mask=mutation_masks[i, :length],
                        wt_sequence=sequences[i],
                        protein_name=names[i],
                        mutation_resolution=mutation_resolutions[i],
                    )
                )

    return None
