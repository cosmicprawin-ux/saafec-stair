#!/usr/bin/env python3
"""Write specified single-mutation DDG predictions from full matrix outputs."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.pipeline_config import work_path_str  # noqa: E402
from core.structure_identifiers import pdb_stem, structure_key  # noqa: E402

AA_ORDER = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION_RE = re.compile(r"^\s*([A-Za-z])\s*(\d+)\s*([A-Za-z])\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract specified single-mutation DDG predictions from predicted_DDG matrices."
    )
    parser.add_argument("--specified-table", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sniff_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        class TabDialect(csv.excel_tab):
            pass

        return TabDialect


def read_table(path: Path) -> list[dict[str, str]]:
    dialect = sniff_dialect(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        return [
            {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def compact_mutation(row: dict[str, str]) -> tuple[str, str, str]:
    mutation = row.get("mutation") or row.get("Mutation") or row.get("mut") or ""
    match = MUTATION_RE.match(mutation)
    if match:
        return match.group(1).upper(), match.group(2), match.group(3).upper()

    wt = (
        row.get("wt_aa")
        or row.get("wild_type_aa")
        or row.get("wildtype")
        or row.get("wild_type")
        or ""
    ).upper()
    pos = (
        row.get("position")
        or row.get("sequence_index")
        or row.get("model_position_1based")
        or row.get("residue")
        or row.get("residue_number")
        or ""
    )
    mt = (
        row.get("mut_aa")
        or row.get("mutant_aa")
        or row.get("mutation_aa")
        or row.get("mutant")
        or ""
    ).upper()
    return wt, pos, mt


def matrix_name_for(row: dict[str, str]) -> str:
    pdb = row.get("pdb") or row.get("pdb_file") or row.get("structure_id") or ""
    chain = row.get("chain") or row.get("chain_id") or ""
    if not pdb or not chain:
        return ""
    return structure_key(pdb, chain)


def load_matrices(matrix_dir: Path) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for path in sorted(matrix_dir.glob("*_predicted_DDG_matrix.csv")):
        protein_name = path.name.removesuffix("_predicted_DDG_matrix.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        by_position = {str(row.get("sequence_index", "")).strip(): row for row in rows}
        matrices[protein_name] = {
            "path": path,
            "fieldnames": fieldnames,
            "by_position": by_position,
        }
    return matrices


def status_row(
    *,
    request: dict[str, str],
    wt: str,
    position: str,
    mt: str,
    predicted: str = "",
    status: str,
    message: str,
    matrix_row: dict[str, str] | None = None,
) -> dict[str, str]:
    matrix_row = matrix_row or {}
    mutation = f"{wt}{position}{mt}" if wt and position and mt else request.get("mutation", "")
    return {
        "pdb": pdb_stem(request.get("pdb") or request.get("pdb_file") or "") if request.get("pdb") or request.get("pdb_file") else "",
        "mutation": mutation,
        "sequence_index": position,
        "chain": matrix_row.get("chain", request.get("chain", "")),
        "pdb_residue_number": matrix_row.get("pdb_residue_number", ""),
        "insertion_code": matrix_row.get("insertion_code", ""),
        "wild_type_aa": wt,
        "mutant_aa": mt,
        "predicted_DDG_kcal_per_mol": predicted,
        "status": status,
        "message": message,
    }


def extract_predictions(specified_rows: list[dict[str, str]], matrices: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for request in specified_rows:
        wt, position, mt = compact_mutation(request)
        matrix_name = matrix_name_for(request)

        if not matrix_name or matrix_name not in matrices:
            output.append(
                status_row(
                    request=request,
                    wt=wt,
                    position=position,
                    mt=mt,
                    status="not_found",
                    message="PDB and chain were not found in predicted_DDG matrices.",
                )
            )
            continue
        if wt not in AA_ORDER or mt not in AA_ORDER or not position:
            output.append(
                status_row(
                    request=request,
                    wt=wt,
                    position=position,
                    mt=mt,
                    status="invalid_request",
                    message="Mutation must specify wild type, 1-based sequence position, and mutant amino acid.",
                )
            )
            continue

        matrix = matrices[matrix_name]
        matrix_row = matrix["by_position"].get(str(position))
        if matrix_row is None:
            output.append(
                status_row(
                    request=request,
                    wt=wt,
                    position=position,
                    mt=mt,
                    status="not_found",
                    message="Sequence position was not found in the predicted_DDG matrix.",
                )
            )
            continue

        matrix_wt = str(matrix_row.get("wild_type_aa", "")).upper()
        if matrix_wt and matrix_wt != wt:
            output.append(
                status_row(
                    request=request,
                    wt=wt,
                    position=position,
                    mt=mt,
                    status="wild_type_mismatch",
                    message=f"Requested wild type {wt}, but matrix has {matrix_wt}.",
                    matrix_row=matrix_row,
                )
            )
            continue

        column = f"{mt}_DDG_kcal_per_mol"
        predicted = str(matrix_row.get(column, "")).strip()
        if not predicted:
            output.append(
                status_row(
                    request=request,
                    wt=wt,
                    position=position,
                    mt=mt,
                    status="not_predicted",
                    message="No DDG prediction is available for this mutation.",
                    matrix_row=matrix_row,
                )
            )
            continue

        output.append(
            status_row(
                request=request,
                wt=wt,
                position=position,
                mt=mt,
                predicted=predicted,
                status="ok",
                message="",
                matrix_row=matrix_row,
            )
        )
    return output


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "pdb",
        "mutation",
        "sequence_index",
        "chain",
        "pdb_residue_number",
        "insertion_code",
        "wild_type_aa",
        "mutant_aa",
        "predicted_DDG_kcal_per_mol",
        "status",
        "message",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    txt_path = csv_path.with_suffix(".txt")
    with csv_path.open(newline="", encoding="utf-8") as src, txt_path.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerows(reader)


def main() -> None:
    args = parse_args()
    specified_table = Path(args.specified_table).expanduser().resolve()
    matrix_dir = Path(args.matrix_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    specified_rows = read_table(specified_table)
    matrices = load_matrices(matrix_dir)
    rows = extract_predictions(specified_rows, matrices)

    csv_path = output_dir / "specified_single_mutation_DDG_predictions.csv"
    write_rows(csv_path, rows)

    summary = {
        "specified_mutation_table": work_path_str(specified_table),
        "matrix_dir": work_path_str(matrix_dir),
        "output_csv": work_path_str(csv_path),
        "output_txt": work_path_str(csv_path.with_suffix(".txt")),
        "n_requested": len(rows),
        "n_predicted": sum(1 for row in rows if row.get("status") == "ok"),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({row.get("status", "") for row in rows})
        },
    }
    (output_dir / "specified_single_mutation_DDG_predictions.summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
