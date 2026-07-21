#!/usr/bin/env python3
"""Create the minimal CSV needed to run full-matrix single inference for one PDB."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.structure_identifiers import pdb_stem, structure_key  # noqa: E402

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


@dataclass(frozen=True)
class ResidueRecord:
    chain: str
    resseq: int
    insertion_code: str
    aa: str

    @property
    def display(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        residue_name = ONE_TO_THREE.get(self.aa, self.aa)
        return f"residue {self.resseq}{suffix} ({residue_name}/{self.aa})"


class PDBCompletenessError(ValueError):
    """Raised when a PDB has unresolved internal residues."""


class PDBInsertionCodeError(ValueError):
    """Raised when protein residues use PDB insertion codes."""


def find_insertion_coded_residues(
    pdb_path: Path, chain: str | None = None
) -> list[tuple[str, int, str, str]]:
    """Return unique standard residues whose PDB insertion-code column is set."""
    residues: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for line in pdb_path.read_text(errors="ignore").splitlines():
        record = line[0:6].strip()
        if record == "ENDMDL":
            break
        if record not in {"ATOM", "HETATM"} or len(line) < 27:
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in THREE_TO_ONE:
            continue
        line_chain = line[21].strip() or "_"
        if chain and line_chain != chain:
            continue
        parsed_resseq = _parse_resseq_token(line[22:26])
        insertion_code = line[26].strip()
        if parsed_resseq is None or not insertion_code:
            continue
        residue = (line_chain, parsed_resseq[0], insertion_code, residue_name)
        if residue not in seen:
            seen.add(residue)
            residues.append(residue)
    return residues


def format_insertion_code_error(
    pdb_path: Path,
    chain: str | None,
    insertion_coded_residues: list[tuple[str, int, str, str]],
) -> str:
    scope = f"chain {chain}" if chain else "all protein chains"
    shown = ", ".join(
        f"{resname} {chain_id}:{resseq}{insertion_code}"
        for chain_id, resseq, insertion_code, resname in insertion_coded_residues[:30]
    )
    if len(insertion_coded_residues) > 30:
        shown += f", ... ({len(insertion_coded_residues)} total)"
    return (
        "Input structure validation failed: PDB insertion codes are not supported for inference.\n"
        f"PDB: {pdb_path.name}\n"
        f"Scope: {scope}\n"
        f"Insertion-coded residues: {shown}\n\n"
        "Insertion-coded protein residue identifiers can make sequence-to-structure and mutation-position "
        "mapping ambiguous. Prediction was not performed. Please provide an input PDB without "
        "insertion-coded protein residues."
    )


def validate_no_insertion_codes(pdb_path: Path, chain: str | None = None) -> None:
    insertion_coded_residues = find_insertion_coded_residues(pdb_path, chain)
    if insertion_coded_residues:
        raise PDBInsertionCodeError(
            format_insertion_code_error(pdb_path, chain, insertion_coded_residues)
        )


def _parse_resseq_token(token: str) -> tuple[int, str] | None:
    token = token.strip()
    if not token:
        return None
    number_chars: list[str] = []
    suffix_chars: list[str] = []
    for idx, char in enumerate(token):
        if char.isdigit() or (idx == 0 and char in {"-", "+"}):
            number_chars.append(char)
        else:
            suffix_chars.append(char)
    if not number_chars:
        return None
    try:
        return int("".join(number_chars)), "".join(suffix_chars).strip()
    except ValueError:
        return None


def parse_remark_465_missing_residues(pdb_path: Path, chain: str | None) -> dict[int, ResidueRecord]:
    missing: dict[int, ResidueRecord] = {}
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("REMARK 465"):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[0] != "REMARK" or parts[1] != "465":
            continue
        residue_name = parts[2].upper()
        if residue_name not in THREE_TO_ONE:
            continue
        line_chain = parts[3]
        parsed_resseq = _parse_resseq_token(parts[4])
        if parsed_resseq is None:
            continue
        if chain and line_chain != chain:
            continue
        resseq, insertion_code = parsed_resseq
        missing[resseq] = ResidueRecord(
            chain=line_chain,
            resseq=resseq,
            insertion_code=insertion_code,
            aa=THREE_TO_ONE[residue_name],
        )
    return missing


def parse_observed_residues(pdb_path: Path, chain: str | None) -> tuple[list[ResidueRecord], set[str]]:
    residues: list[ResidueRecord] = []
    seen: set[tuple[str, str, str]] = set()
    chains_seen: set[str] = set()
    for line in pdb_path.read_text(errors="ignore").splitlines():
        record = line[0:6].strip()
        if record == "ENDMDL":
            break
        if record != "ATOM":
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A", "1"}:
            continue
        line_chain = line[21].strip() or "_"
        chains_seen.add(line_chain)
        if chain and line_chain != chain:
            continue
        residue_name = line[17:20].strip().upper()
        aa = THREE_TO_ONE.get(residue_name)
        if aa is None:
            continue
        parsed_resseq = _parse_resseq_token(line[22:26].strip())
        if parsed_resseq is None:
            continue
        resseq, parsed_insertion_code = parsed_resseq
        insertion_code = line[26].strip() or parsed_insertion_code
        key = (line_chain, str(resseq), insertion_code)
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            ResidueRecord(
                chain=line_chain,
                resseq=resseq,
                insertion_code=insertion_code,
                aa=aa,
            )
        )
    return residues, chains_seen


def find_internal_missing_residues(pdb_path: Path, chain: str | None) -> list[ResidueRecord]:
    observed_residues, chains_seen = parse_observed_residues(pdb_path, chain)
    if not observed_residues:
        chain_msg = f" for chain {chain}" if chain else ""
        raise ValueError(f"No standard amino-acid ATOM residues found in {pdb_path}{chain_msg}. Chains seen: {sorted(chains_seen)}")

    missing_from_remarks = parse_remark_465_missing_residues(pdb_path, chain)
    internal_missing: dict[int, ResidueRecord] = {}

    by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in observed_residues:
        by_chain.setdefault(residue.chain, []).append(residue)

    for chain_id, residues in by_chain.items():
        ordered = sorted(residues, key=lambda item: (item.resseq, item.insertion_code))
        observed_numbers = [residue.resseq for residue in ordered]
        if len(observed_numbers) < 2:
            continue
        min_observed = min(observed_numbers)
        max_observed = max(observed_numbers)
        observed_set = set(observed_numbers)
        for resseq in range(min_observed + 1, max_observed):
            if resseq in observed_set:
                continue
            remark_record = missing_from_remarks.get(resseq)
            if remark_record is not None:
                internal_missing[resseq] = remark_record
            else:
                internal_missing[resseq] = ResidueRecord(
                    chain=chain_id,
                    resseq=resseq,
                    insertion_code="",
                    aa="unknown residue",
                )

    return [internal_missing[key] for key in sorted(internal_missing)]


def format_missing_residue_error(pdb_path: Path, chain: str | None, missing_residues: list[ResidueRecord]) -> str:
    chain_msg = f"chain {chain}" if chain else "selected chain(s)"
    grouped: dict[str, list[ResidueRecord]] = {}
    for residue in missing_residues[:30]:
        grouped.setdefault(residue.chain, []).append(residue)
    shown = "; ".join(
        f"chain {chain_id}: " + ", ".join(residue.display for residue in residues)
        for chain_id, residues in grouped.items()
    )
    if len(missing_residues) > 30:
        shown += f", ... ({len(missing_residues)} total)"
    return (
        "Input structure validation failed: unresolved internal residues were detected in the input PDB.\n"
        f"PDB: {pdb_path.name}\n"
        f"Scope: {chain_msg}\n"
        f"Internal missing residues: {shown}\n\n"
        "Prediction was not performed because this pipeline requires a complete backbone for the modeled "
        "protein segment. Using only the observed residues would alter the protein representation, and "
        "inserting gap or zero-valued rows would provide unsupported inputs to SaProt, ProteinMPNN, and "
        "local-contact features. Please provide a PDB/model in which these residues are resolved, or use a "
        "structure without internal missing residues for this prediction."
    )


def parse_sequence(pdb_path: Path, chain: str | None) -> str:
    observed_residues, chains_seen = parse_observed_residues(pdb_path, chain)
    if not observed_residues:
        chain_msg = f" for chain {chain}" if chain else ""
        raise ValueError(f"No standard amino-acid ATOM residues found in {pdb_path}{chain_msg}. Chains seen: {sorted(chains_seen)}")
    return "".join(residue.aa for residue in observed_residues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pdb-output-dir", required=True)
    args = parser.parse_args()

    pdb_path = Path(args.pdb).expanduser().resolve()
    pdb = pdb_stem(pdb_path.name)
    protein_name = structure_key(pdb, args.chain)
    pdb_output_dir = Path(args.pdb_output_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    pdb_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    validate_no_insertion_codes(pdb_path, args.chain)

    missing_residues = find_internal_missing_residues(pdb_path, args.chain)
    if missing_residues:
        raise PDBCompletenessError(format_missing_residue_error(pdb_path, args.chain, missing_residues))

    sequence = parse_sequence(pdb_path, args.chain)
    wt = sequence[0]
    mt = "A" if wt != "A" else "C"
    shutil.copy2(pdb_path, pdb_output_dir / f"{pdb}.pdb")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pdb", "chain", "position", "wt_aa", "mut_aa"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"pdb": pdb, "chain": args.chain, "position": 1, "wt_aa": wt, "mut_aa": mt})
    print(f"Wrote {output_csv} for {protein_name} ({len(sequence)} residues).")


if __name__ == "__main__":
    try:
        main()
    except (PDBCompletenessError, PDBInsertionCodeError) as exc:
        raise SystemExit(str(exc))
