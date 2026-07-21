"""All-mutant matrix and HTML heatmap export for inference results."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import torch

from core.pipeline_config import work_path_str
from core.amino_acids import AMINO_ACIDS_20


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
DELTA_DELTA_G_COLUMN_LABEL = "DDG"
CSV_PREFIX_COLUMNS = [
    "sequence_index",
    "chain",
    "pdb_residue_number",
    "insertion_code",
    "wild_type_aa",
    "wild_type_residue",
]

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_pdb_residues(pdb_path: Path) -> list[dict[str, str]]:
    residues: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for line in pdb_path.read_text(errors="ignore").splitlines():
        record = line[0:6].strip()
        if record == "ENDMDL":
            break
        if record != "ATOM":
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A", "1"}:
            continue
        residue_name = line[17:20].strip().upper()
        wild_type = THREE_TO_ONE.get(residue_name)
        if wild_type is None:
            continue
        chain = line[21].strip() or "_"
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "sequence_index": str(len(residues) + 1),
                "chain": chain,
                "pdb_residue_number": residue_number,
                "insertion_code": insertion_code,
                "wild_type_aa": wild_type,
                "wild_type_residue": residue_name,
            }
        )
    return residues


def fallback_residues(sequence: str) -> list[dict[str, str]]:
    return [
        {
            "sequence_index": str(index),
            "chain": "_",
            "pdb_residue_number": str(index),
            "insertion_code": "",
            "wild_type_aa": aa,
            "wild_type_residue": aa,
        }
        for index, aa in enumerate(sequence, start=1)
    ]


def find_pdb_for_protein(protein_name: str, pdb_dir: Path | None) -> Path | None:
    if pdb_dir is None or not pdb_dir.is_dir():
        return None
    candidates = [
        pdb_dir / f"{protein_name}.pdb",
        pdb_dir / f"{protein_name.upper()}.pdb",
        pdb_dir / f"{protein_name.lower()}.pdb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    lower_name = protein_name.lower()
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    for candidate in pdb_files:
        if candidate.stem.lower() == lower_name:
            return candidate
    prefix_matches = [
        candidate
        for candidate in pdb_files
        if lower_name.startswith(f"{candidate.stem.lower()}_")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def matrix_rows_for_protein(
    *,
    sequence: str,
    predicted_ddg_matrix: torch.Tensor,
    pdb_path: Path | None,
) -> list[dict[str, Any]]:
    residues = parse_pdb_residues(pdb_path) if pdb_path is not None and pdb_path.is_file() else []
    if len(residues) != len(sequence):
        residues = fallback_residues(sequence)

    aa_to_index = {aa: idx for idx, aa in enumerate(AMINO_ACIDS_20)}
    rows: list[dict[str, Any]] = []
    for pos_idx, wt_aa in enumerate(sequence):
        residue = dict(residues[pos_idx])
        residue["wild_type_aa"] = wt_aa
        row: dict[str, Any] = {column: residue.get(column, "") for column in CSV_PREFIX_COLUMNS}
        row["sequence_index"] = str(pos_idx + 1)
        for aa in AA_ORDER:
            if aa == wt_aa:
                value = 0.0
            else:
                aa_idx = aa_to_index[aa]
                value = float(predicted_ddg_matrix[pos_idx, aa_idx].item())
            row[f"{aa}_{DELTA_DELTA_G_COLUMN_LABEL}_kcal_per_mol"] = f"{value:.8g}"
        rows.append(row)
    return rows


def write_all_mutation_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CSV_PREFIX_COLUMNS + [f"{aa}_{DELTA_DELTA_G_COLUMN_LABEL}_kcal_per_mol" for aa in AA_ORDER]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_txt_copy(csv_path: Path) -> Path:
    txt_path = csv_path.with_suffix(".txt")
    with csv_path.open(newline="", encoding="utf-8") as src, txt_path.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerows(reader)
    return txt_path


def write_visualization_outputs(
    *,
    output_dir: Path,
    ensemble_matrices: dict[str, dict[str, Any]],
    pdb_dir: Path | None,
    write_visualizations: bool,
    template_path: Path | None,
) -> list[dict[str, Any]]:
    visualization_records: list[dict[str, Any]] = []
    if write_visualizations:
        from visualization.create_ddg_heatmap import read_matrix, read_structure, render_html  # noqa: WPS433

    for protein_name, entry in ensemble_matrices.items():
        protein_dir = output_dir / "visualizations" / protein_name
        matrix_path = protein_dir / f"{protein_name}_all_mutation_ΔΔG_matrix.csv"
        predicted_ddg_dir = output_dir / "predicted_DDG"
        predicted_ddg_path = predicted_ddg_dir / f"{protein_name}_predicted_DDG_matrix.csv"
        html_path = protein_dir / f"{protein_name}_ΔΔG_heatmap.html"
        pdb_path = find_pdb_for_protein(protein_name, pdb_dir)
        matrix_rows = matrix_rows_for_protein(
            sequence=entry["sequence"],
            predicted_ddg_matrix=entry["predicted_ddg_matrix"],
            pdb_path=pdb_path,
        )
        write_all_mutation_matrix_csv(matrix_path, matrix_rows)
        predicted_ddg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matrix_path, predicted_ddg_path)
        predicted_ddg_txt_path = write_txt_copy(predicted_ddg_path)

        record = {
            "structure_key": protein_name,
            "residue_count": len(entry["sequence"]),
            "all_mutation_count": len(entry["sequence"]) * (len(AA_ORDER) - 1),
            "matrix_csv": work_path_str(matrix_path),
            "predicted_DDG_csv": work_path_str(predicted_ddg_path),
            "predicted_DDG_txt": work_path_str(predicted_ddg_txt_path),
            "pdb_path": work_path_str(pdb_path) if pdb_path is not None else None,
            "html": None,
        }
        if write_visualizations:
            resolved_template = template_path or (SCRIPTS_ROOT / "visualization" / "ddg_heatmap.html")
            rows, meta = read_matrix(matrix_path)
            structure = read_structure(pdb_path, rows)
            title = f"{protein_name} Mutation ΔΔG Stability Predictions"
            html_path.write_text(
                render_html(rows, meta, title, structure, resolved_template),
                encoding="utf-8",
            )
            record["html"] = work_path_str(html_path)
            record["html_mutation_count"] = meta["mutationCount"]
        visualization_records.append(record)
    return visualization_records
