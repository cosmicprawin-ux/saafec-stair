#!/usr/bin/env python3
"""SAAFEC-native PDB parsing helpers with validated mutation-index behavior.

The parser is self-contained and does not import or execute external parser
source.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


AA3_TO_1 = {
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
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "HYP": "P",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "CSO": "C",
    "CME": "C",
    "CSD": "C",
    "KCX": "K",
}


def first_atom_chain(pdb_path: Path) -> str | None:
    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                return line[21].strip() or None
    return None


def safe_first_atom_chain(pdb_path: Path) -> str | None:
    try:
        return first_atom_chain(pdb_path)
    except Exception:
        return None


def infer_chain_from_stem(stem: str) -> str | None:
    if "_" not in stem:
        return None
    candidate = stem.rsplit("_", 1)[-1]
    return candidate if len(candidate) == 1 else None


def unique_chain_candidates(*candidates: str | None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def _atom_residue_record(line: str) -> tuple[str, str, str, str] | None:
    if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
        return None
    residue_name = line[17:20].strip().upper()
    amino_acid = AA3_TO_1.get(residue_name)
    if amino_acid is None:
        return None
    chain = line[21].strip()
    residue_number = line[22:26].strip()
    insertion_code = line[26].strip()
    if not residue_number:
        return None
    return chain, residue_number, insertion_code, amino_acid


def parse_pdb_chain(pdb_path: Path, chain: str) -> dict[str, Any]:
    """Parse one PDB chain into one-letter sequence and residue-number list."""
    sequence: list[str] = []
    residue_numbers: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = _atom_residue_record(line)
            if record is None:
                continue
            record_chain, residue_number, insertion_code, amino_acid = record
            if record_chain != chain:
                continue
            residue_key = (record_chain, residue_number, insertion_code)
            if residue_key in seen:
                continue
            seen.add(residue_key)
            sequence.append(amino_acid)
            residue_numbers.append(residue_number)
    if not sequence:
        raise ValueError(f"No amino-acid residues found for chain {chain!r} in {pdb_path}")
    return {"seq": "".join(sequence), "resn_list": residue_numbers}


def build_structure_parser_context(
    *,
    pdb_path: Path,
    protein_name: str,
    chain_candidates: list[str],
) -> dict[str, Any]:
    """Parse a protein and return sequence context for mutation indexing."""
    errors: list[dict[str, str]] = []
    for chain in chain_candidates:
        try:
            raw_pdb = parse_pdb_chain(pdb_path, chain)
            seq = str(raw_pdb.get("seq", ""))
            resn_list = [str(item) for item in raw_pdb.get("resn_list", [])]
            return {
                "protein_name": protein_name,
                "pdb_path": str(pdb_path),
                "chain_used": chain,
                "seq": seq,
                "resn_list": resn_list,
                "parser": "saafec_native_pdb_chain_parser",
            }
        except Exception as exc:  # noqa: BLE001 - preserve chain probing.
            errors.append({"chain": chain, "error": repr(exc)})
    raise RuntimeError(
        "Native PDB chain parser failed for all chain candidates "
        f"for {protein_name} at {pdb_path}: {errors}"
    )


def sequence_indexed_dataset(dataset_name: str) -> bool:
    name = dataset_name.lower()
    return (
        name.startswith("domainome")
        or name.startswith("megascale")
        or name.startswith("fireprot")
        or name.startswith("full_fireprot")
    )


def resolve_sequence_index(mutation: dict[str, Any], seq: str) -> tuple[int | None, str]:
    wt = mutation["wt_aa"]
    pos_seq = mutation.get("position_seq") or mutation["position_pdb"]
    idx = int(pos_seq) - 1
    if idx < 0 or idx >= len(seq):
        return None, "reference_sequence_position_out_of_range"
    if seq[idx] == wt:
        return idx, "reference_sequence_one_based"
    return None, f"reference_sequence_wt_mismatch:saw_{seq[idx]}_expected_{wt}"


def resolve_resn_list_index(
    mutation: dict[str, Any],
    seq: str,
    resn_list: list[str],
    dataset_name: str,
    pdb_path: str | Path,
) -> tuple[int | None, str]:
    wt = mutation["wt_aa"]
    pos = str(mutation["position_pdb"])
    try:
        pdb_idx = resn_list.index(pos)
    except ValueError:
        return None, "reference_residue_number_not_found"

    if pdb_idx < len(seq) and seq[pdb_idx] == wt:
        return pdb_idx, "reference_residue_number_index"

    if "s669" in dataset_name.lower() or "S669" in str(pdb_path):
        gaps = [aa for aa in seq if aa == "-"]
    else:
        gaps = [aa for aa in seq[: pdb_idx + 10] if aa == "-"]
    adjusted_idx = pdb_idx + len(gaps) if gaps else pdb_idx + 1
    if 0 <= adjusted_idx < len(seq) and seq[adjusted_idx] == wt:
        return adjusted_idx, "reference_gap_adjusted_index"
    saw = seq[adjusted_idx] if 0 <= adjusted_idx < len(seq) else "OUT_OF_RANGE"
    return None, f"reference_wt_mismatch:saw_{saw}_expected_{wt}"


def _normalise_dataset_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "unknown"


def mutation_dataset_name(mutation: dict[str, Any], default_dataset_name: str) -> str:
    source_database = _normalise_dataset_name(mutation.get("source_database"))
    if source_database and source_database != "unknown":
        return source_database
    return _normalise_dataset_name(default_dataset_name)


def resolve_reference_mutation_index(
    mutation: dict[str, Any],
    *,
    parser_context: dict[str, Any],
    default_dataset_name: str,
) -> tuple[int | None, str]:
    """Resolve one mutation against the parsed reference chain."""
    dataset_name = mutation_dataset_name(mutation, default_dataset_name)
    seq = str(parser_context.get("seq", ""))
    resn_list = [str(item) for item in parser_context.get("resn_list", [])]
    if sequence_indexed_dataset(dataset_name):
        return resolve_sequence_index(mutation, seq)
    return resolve_resn_list_index(
        mutation,
        seq,
        resn_list,
        dataset_name,
        parser_context.get("pdb_path", ""),
    )
