#!/usr/bin/env python3
"""Generate frozen ProteinMPNN residue-logit caches on a reference sequence axis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from cache.sequence_variant_policy import choose_canonical_sequence  # noqa: E402
from core.mutation_dataset import (  # noqa: E402
    _global_align_reference_to_model,
    _normalise_column_name,
    load_protein_names_csv,
    load_workbook_records_xlsx,
    resolve_dataset_csv_path,
)
from core.pipeline_config import WORK_DIR, ensure_output_root, work_path_str  # noqa: E402
from core.pipeline_config import resolve_output_path  # noqa: E402
from core.SAAFEC_STAIR_pdb_chain_parser import (  # noqa: E402
    AA3_TO_1,
    infer_chain_from_stem,
    safe_first_atom_chain,
    unique_chain_candidates,
)
from external.proteinmpnn_loader import load_proteinmpnn_class  # noqa: E402
from core.amino_acids import AMINO_ACIDS_20  # noqa: E402
from core.structure_identifiers import pdb_stem, structure_key  # noqa: E402


STANDARD_AA = set(AMINO_ACIDS_20)
BACKBONE_ATOM_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3}
PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_PROTEINMPNN_CHECKPOINT = "assets/external/models/proteinmpnn/v_48_020.pt"
DEFAULT_PROTEINMPNN_SOURCE = "assets/external/source/ThermoMPNN/protein_mpnn_utils.py"


@dataclass(frozen=True)
class ProteinMPNNCacheJob:
    protein_name: str
    structure_id: str
    chain_id: str | None
    sequence: str
    sequence_source: str
    canonical_sequence_policy: str
    sequence_variant_summary: tuple[dict[str, Any], ...]
    pdb_path: Path | None = None


@dataclass(frozen=True)
class BackboneStructureInput:
    sequence: str
    coordinates: torch.Tensor
    coordinate_mask: torch.Tensor
    chain_id: str
    sequence_source: str
    residue_records: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen ProteinMPNN logits aligned to a reference cache sequence "
            "(SaProt), then workbook sequence, then PDB fallback."
        )
    )
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--table-sheet", dest="table_sheet", default="refined_sorted_clean")
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-cache-dir",
        default=None,
        help=(
            "Optional by_protein cache whose metadata.json sequence defines the "
            "target residue axis, e.g. the SaProt embedding cache."
        ),
    )
    parser.add_argument("--proteinmpnn-checkpoint", default=DEFAULT_PROTEINMPNN_CHECKPOINT)
    parser.add_argument(
        "--proteinmpnn-source",
        default=DEFAULT_PROTEINMPNN_SOURCE,
        help="Path to the pinned upstream protein_mpnn_utils.py module.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def clean_sequence(value: Any) -> str:
    if value is None:
        return ""
    return "".join(str(value).upper().split())


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_existing_path(raw_path: str | Path, *, label: str) -> Path:
    raw = Path(raw_path).expanduser()
    candidates = [
        raw if raw.is_absolute() else WORK_DIR / raw,
        raw.resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} not found: {raw_path}")


def _resolve_optional_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalised_to_actual = {
        _normalise_column_name(name): name
        for name in fieldnames
    }
    for candidate in candidates:
        actual = normalised_to_actual.get(_normalise_column_name(candidate))
        if actual is not None:
            return actual
    return None


def _table_delimiter(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    return "\t" if "\t" in header else ","


def load_proteinmpnn(
    *,
    checkpoint_path: str | Path,
    source_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    ProteinMPNN = load_proteinmpnn_class(source_path)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        augment_eps=0.0,
    ).to(device)
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ProteinMPNN checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    matched = sorted(set(state).intersection(model.state_dict()))
    if not matched:
        raise RuntimeError(f"ProteinMPNN checkpoint did not match model state keys: {checkpoint_path}")
    print(
        f"Loaded ProteinMPNN checkpoint {work_path_str(ckpt_path)} "
        f"(matched={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def sequence_jobs_from_xlsx(table_path: Path, sheet_name: str) -> list[ProteinMPNNCacheJob]:
    records = load_workbook_records_xlsx(table_path, sheet_name=sheet_name)
    jobs: list[ProteinMPNNCacheJob] = []
    for protein_name, record in sorted(records.items()):
        sequence_counts = Counter(
            clean_sequence(mutation.get("wt_sequence"))
            for mutation in record.mutations
            if clean_sequence(mutation.get("wt_sequence"))
        )
        if sequence_counts:
            sequence, policy, variant_summary = choose_canonical_sequence(
                protein_name,
                sequence_counts,
                table_path,
            )
            sequence_source = (
                "workbook_wt_seq_pdb_canonical"
                if len(sequence_counts) > 1
                else "workbook_wt_seq_pdb"
            )
        else:
            sequence = ""
            policy = "missing_workbook_wt_sequence_pending_pdb_fallback"
            variant_summary = ()
            sequence_source = "missing_workbook_wt_sequence"
        jobs.append(
            ProteinMPNNCacheJob(
                protein_name=protein_name,
                structure_id=record.structure_id,
                chain_id=record.chain,
                sequence=sequence,
                sequence_source=sequence_source,
                canonical_sequence_policy=policy,
                sequence_variant_summary=variant_summary,
            )
        )
    return jobs


def sequence_jobs_from_csv(table_path: Path) -> list[ProteinMPNNCacheJob]:
    with table_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=_table_delimiter(table_path))
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {table_path}")
        pdb_col = _resolve_optional_column(reader.fieldnames, ["pdb", "pdb_file", "structure", "structure_id"])
        chain_col = _resolve_optional_column(reader.fieldnames, ["chain", "chain_id"])
        sequence_col = _resolve_optional_column(
            reader.fieldnames,
            ["wt_sequence", "wt_seq", "sequence", "wildtype_sequence"],
        )
        if pdb_col is None or chain_col is None:
            load_protein_names_csv(table_path)
            raise ValueError(f"Could not resolve PDB and chain columns in {table_path}: {reader.fieldnames}")
        by_name: dict[str, Counter[str]] = {}
        structure_by_name: dict[str, str] = {}
        chain_by_name: dict[str, str] = {}
        for row in reader:
            pdb = str(row.get(pdb_col, "")).strip()
            chain = str(row.get(chain_col, "")).strip()
            if not pdb or not chain:
                continue
            protein_name = structure_key(pdb, chain)
            by_name.setdefault(protein_name, Counter())
            structure_by_name[protein_name] = pdb_stem(pdb)
            chain_by_name[protein_name] = chain
            if sequence_col is not None:
                sequence = clean_sequence(row.get(sequence_col))
                if sequence:
                    by_name[protein_name][sequence] += 1

    jobs: list[ProteinMPNNCacheJob] = []
    for protein_name, sequence_counts in sorted(by_name.items()):
        if sequence_counts:
            sequence, policy, variant_summary = choose_canonical_sequence(
                protein_name,
                sequence_counts,
                table_path,
            )
            sequence_source = (
                "csv_wt_sequence_canonical"
                if len(sequence_counts) > 1
                else "csv_wt_sequence"
            )
        else:
            sequence = ""
            policy = "missing_csv_wt_sequence_pending_pdb_fallback"
            variant_summary = ()
            sequence_source = "missing_csv_wt_sequence"
        jobs.append(
            ProteinMPNNCacheJob(
                protein_name=protein_name,
                structure_id=structure_by_name[protein_name],
                chain_id=chain_by_name[protein_name],
                sequence=sequence,
                sequence_source=sequence_source,
                canonical_sequence_policy=policy,
                sequence_variant_summary=variant_summary,
            )
        )
    return jobs


def load_sequence_jobs(dataset_csv: str | Path, sheet_name: str) -> tuple[Path, list[ProteinMPNNCacheJob]]:
    resolved = resolve_dataset_csv_path(dataset_csv)
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset table not found: {resolved}")
    if resolved.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return resolved, sequence_jobs_from_xlsx(resolved, sheet_name)
    return resolved, sequence_jobs_from_csv(resolved)


def resolve_reference_cache_dir(raw_path: str | Path) -> Path:
    raw = Path(raw_path).expanduser()
    candidates = [
        raw if raw.is_absolute() else WORK_DIR / raw,
        resolve_output_path(raw),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            by_protein = candidate / "by_protein"
            return by_protein if by_protein.is_dir() else candidate
    raise FileNotFoundError(f"Reference sequence cache directory not found: {raw_path}")


def apply_reference_cache_sequences(
    jobs: list[ProteinMPNNCacheJob],
    reference_cache_dir: str | Path | None,
) -> tuple[list[ProteinMPNNCacheJob], list[dict[str, str]]]:
    if reference_cache_dir is None:
        return jobs, []
    root = resolve_reference_cache_dir(reference_cache_dir)
    updated: list[ProteinMPNNCacheJob] = []
    missing: list[dict[str, str]] = []
    for job in jobs:
        metadata_path = root / job.protein_name / "metadata.json"
        if not metadata_path.is_file():
            missing.append({"protein_name": job.protein_name, "metadata_path": work_path_str(metadata_path)})
            updated.append(job)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sequence = clean_sequence(metadata.get("sequence"))
        if not sequence:
            missing.append({"protein_name": job.protein_name, "metadata_path": work_path_str(metadata_path)})
            updated.append(job)
            continue
        updated.append(
            replace(
                job,
                sequence=sequence,
                sequence_source=f"reference_cache:{metadata.get('sequence_source', 'metadata_sequence')}",
                canonical_sequence_policy="reference_cache_sequence_preferred",
                sequence_variant_summary=(
                    {
                        "sequence": sequence,
                        "length": len(sequence),
                        "count": 1,
                        "source": work_path_str(metadata_path),
                    },
                ),
            )
        )
    return updated, missing


def attach_pdb_paths(
    jobs: list[ProteinMPNNCacheJob],
    pdb_dir: str | Path,
) -> tuple[list[ProteinMPNNCacheJob], list[dict[str, str]]]:
    pdb_root = resolve_existing_path(pdb_dir, label="PDB directory")
    available = {path.stem: path.resolve() for path in sorted(pdb_root.glob("*.pdb"))}
    available_lower = {stem.lower(): path for stem, path in available.items()}
    if not available:
        raise FileNotFoundError(f"No .pdb files found in PDB directory: {pdb_root}")

    resolved_jobs: list[ProteinMPNNCacheJob] = []
    missing: list[dict[str, str]] = []
    for job in jobs:
        pdb_path = (
            available.get(job.structure_id)
            or available_lower.get(job.structure_id.lower())
            or available.get(job.protein_name)
            or available_lower.get(job.protein_name.lower())
        )
        if pdb_path is None:
            missing.append({"protein_name": job.protein_name, "structure_id": job.structure_id})
            continue
        resolved_jobs.append(replace(job, pdb_path=pdb_path))
    if not resolved_jobs:
        raise FileNotFoundError(f"No matching PDB files found in {pdb_root}.")
    return resolved_jobs, missing


def _parse_backbone_for_chain(pdb_path: Path, chain_id: str) -> BackboneStructureInput:
    records_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[int, str]] = []

    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            if len(line) <= 21 or line[21].strip() != chain_id.strip():
                continue
            resname = line[17:20].strip().upper()
            if line.startswith("HETATM") and resname != "MSE":
                continue
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            ins_code = line[26].strip()
            atom_idx = BACKBONE_ATOM_ORDER.get(line[12:16].strip())
            if atom_idx is None:
                continue
            try:
                xyz = torch.tensor(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=torch.float32,
                )
            except ValueError:
                continue

            key = (resseq, ins_code)
            if key not in records_by_key:
                ordered_keys.append(key)
                records_by_key[key] = {
                    "aa": aa,
                    "coordinates": torch.full((4, 3), float("nan"), dtype=torch.float32),
                    "pdb_resseq": resseq,
                    "pdb_ins_code": ins_code,
                }
            records_by_key[key]["coordinates"][atom_idx] = xyz

    if not ordered_keys:
        raise ValueError(f"No amino-acid backbone atoms found for chain {chain_id!r} in {pdb_path}")

    sequence = "".join(str(records_by_key[key]["aa"]) for key in ordered_keys)
    coordinates = torch.stack([records_by_key[key]["coordinates"] for key in ordered_keys], dim=0)
    coordinate_mask = torch.isfinite(coordinates).all(dim=(-1, -2))
    residue_records = tuple(
        {
            "observed_index": idx,
            "aa": records_by_key[key]["aa"],
            "chain_id": chain_id,
            "pdb_resseq": records_by_key[key]["pdb_resseq"],
            "pdb_ins_code": records_by_key[key]["pdb_ins_code"],
            "has_backbone_n_ca_c_o": bool(coordinate_mask[idx - 1].item()),
        }
        for idx, key in enumerate(ordered_keys, start=1)
    )
    return BackboneStructureInput(
        sequence=sequence,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        chain_id=chain_id,
        sequence_source="pdb_atom_records_backbone",
        residue_records=residue_records,
    )


def parse_backbone_structure_input(
    *,
    pdb_path: Path,
    protein_name: str,
    structure_id: str,
    chain_id: str | None,
) -> BackboneStructureInput:
    chain_candidates = unique_chain_candidates(
        chain_id,
        infer_chain_from_stem(protein_name),
        infer_chain_from_stem(structure_id),
        infer_chain_from_stem(pdb_path.stem),
        safe_first_atom_chain(pdb_path),
        "A",
    )
    if not chain_candidates:
        chain_candidates = [""]

    errors: list[dict[str, str]] = []
    for candidate in chain_candidates:
        try:
            return _parse_backbone_for_chain(pdb_path, candidate)
        except Exception as exc:
            errors.append({"chain": candidate, "error": repr(exc)})
    raise RuntimeError(
        f"ProteinMPNN backbone parser failed for all chain candidates for "
        f"{protein_name} at {pdb_path}: {errors}"
    )


def align_coordinates_to_sequence(
    *,
    full_sequence: str,
    full_coordinates: torch.Tensor,
    target_sequence: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not target_sequence:
        target_sequence = full_sequence
    length = len(target_sequence)
    coords = torch.full((length, 4, 3), float("nan"), dtype=torch.float32)
    source_to_target = _global_align_reference_to_model(full_sequence, target_sequence)
    assigned = 0
    mismatches: list[dict[str, Any]] = []
    for source_idx, target_idx in enumerate(source_to_target):
        if target_idx is None or target_idx < 0 or target_idx >= length:
            continue
        source_aa = full_sequence[source_idx]
        target_aa = target_sequence[target_idx]
        compatible = (
            source_aa == target_aa
            or source_aa in {"X", "B", "Z", "U", "O"}
            or target_aa in {"X", "B", "Z", "U", "O"}
        )
        if not compatible:
            mismatches.append(
                {
                    "source_index": source_idx + 1,
                    "target_index": target_idx + 1,
                    "source_aa": source_aa,
                    "target_aa": target_aa,
                }
            )
            continue
        residue_coords = full_coordinates[source_idx, :4, :].float()
        coords[target_idx] = residue_coords
        assigned += int(torch.isfinite(residue_coords).all().item())
    mask = torch.isfinite(coords).all(dim=(-1, -2))
    summary = {
        "source_sequence_length": len(full_sequence),
        "target_sequence_length": length,
        "mapped_coordinate_rows": int(mask.sum().item()),
        "assigned_rows_before_finite_check": assigned,
        "alignment_mismatches": mismatches[:25],
        "alignment_mismatch_count": len(mismatches),
    }
    return coords, mask, summary


@torch.no_grad()
def run_proteinmpnn_logits(
    model: torch.nn.Module,
    *,
    sequence: str,
    coordinates: torch.Tensor,
    coordinate_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 3):
        raise ValueError(f"Expected coordinates with shape (L,4,3), got {tuple(coordinates.shape)}")
    if len(sequence) != coordinates.shape[0]:
        raise ValueError(f"Sequence length {len(sequence)} does not match coordinates length {coordinates.shape[0]}")
    x = torch.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0).to(device)
    mask = coordinate_mask.float().unsqueeze(0).to(device)
    x_index = PROTEINMPNN_ALPHABET.index("X")
    sequence_ids = torch.tensor(
        [
            [
                PROTEINMPNN_ALPHABET.index(aa)
                if aa in PROTEINMPNN_ALPHABET
                else x_index
                for aa in sequence
            ]
        ],
        dtype=torch.long,
        device=device,
    )
    residue_idx = torch.arange(len(sequence), dtype=torch.long, device=device).unsqueeze(0)
    chain_encoding_all = torch.ones_like(residue_idx)
    randn = torch.zeros_like(mask)
    outputs = model(x, sequence_ids, mask, mask.clone(), residue_idx, chain_encoding_all, randn, False)
    logits_21 = outputs[-1] if isinstance(outputs, tuple) else outputs
    if logits_21.ndim != 3:
        raise ValueError(f"ProteinMPNN returned logits with unexpected shape {tuple(logits_21.shape)}")
    canonical_indices = [PROTEINMPNN_ALPHABET.index(aa) for aa in AMINO_ACIDS_20]
    logits = logits_21[0, :, canonical_indices].detach().float().cpu().contiguous()
    return logits * coordinate_mask.detach().cpu().float().unsqueeze(-1)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "protein_name",
        "structure_id",
        "pdb_path",
        "chain_id",
        "sequence_length",
        "coordinate_mask_true",
        "sequence_sha",
        "proteinmpnn_checkpoint_sha",
        "sequence_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = ensure_output_root(args.output_dir)
    by_protein_root = output_root / "by_protein"
    by_protein_root.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    device = choose_device(args.device)
    resolved_dataset, loaded_jobs = load_sequence_jobs(args.dataset_csv, args.table_sheet)
    sequence_jobs, reference_sequence_missing = apply_reference_cache_sequences(
        loaded_jobs,
        args.reference_cache_dir,
    )
    jobs, missing_pdbs = attach_pdb_paths(sequence_jobs, args.pdb_dir)
    model = load_proteinmpnn(
        checkpoint_path=args.proteinmpnn_checkpoint,
        source_path=args.proteinmpnn_source,
        device=device,
    )
    checkpoint_sha = sha256_of_file(Path(args.proteinmpnn_checkpoint).expanduser().resolve())
    source_sha = sha256_of_file(Path(args.proteinmpnn_source).expanduser().resolve())
    manifest_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for job in jobs:
        protein_start = perf_counter()
        print(f"[run ] {job.protein_name}")
        try:
            if job.pdb_path is None:
                raise ValueError("Internal error: PDB path is missing after attachment.")
            backbone = parse_backbone_structure_input(
                pdb_path=job.pdb_path,
                protein_name=job.protein_name,
                structure_id=job.structure_id,
                chain_id=job.chain_id,
            )
            target_sequence = job.sequence or backbone.sequence
            sequence_source = job.sequence_source
            if not job.sequence:
                sequence_source = "pdb_atom_record_backbone_fallback_no_workbook_sequence"
            proteinmpnn_sequence = "".join(
                aa if aa in PROTEINMPNN_ALPHABET else "X"
                for aa in target_sequence
            )
            coordinates, coordinate_mask, alignment_summary = align_coordinates_to_sequence(
                full_sequence=backbone.sequence,
                full_coordinates=backbone.coordinates,
                target_sequence=target_sequence,
            )
            logits = run_proteinmpnn_logits(
                model,
                sequence=proteinmpnn_sequence,
                coordinates=coordinates,
                coordinate_mask=coordinate_mask,
                device=device,
            )
            if logits.shape != (len(target_sequence), 20):
                raise ValueError(
                    f"ProteinMPNN logits shape {tuple(logits.shape)} does not match "
                    f"target sequence length {len(target_sequence)}."
                )
            nonstandard_counts = {
                aa: target_sequence.count(aa)
                for aa in sorted(set(target_sequence) - STANDARD_AA)
            }
            protein_dir = by_protein_root / job.protein_name
            protein_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "protein_name": job.protein_name,
                "structure_id": job.structure_id,
                "pdb_path": work_path_str(job.pdb_path),
                "chain_id": backbone.chain_id,
                "requested_chain_id": job.chain_id,
                "sequence": target_sequence,
                "proteinmpnn_sequence": proteinmpnn_sequence,
                "sequence_source": sequence_source,
                "canonical_sequence_policy": job.canonical_sequence_policy,
                "wt_sequence_variants": list(job.sequence_variant_summary),
                "length": len(target_sequence),
                "nonstandard_residue_counts": nonstandard_counts,
                "nonstandard_residue_policy": (
                    "Residues unsupported by ProteinMPNN are mapped to X for the "
                    "frozen ProteinMPNN sequence channel; the DDG head predicts only "
                    "20 standard mutant channels."
                ),
                "pdb_backbone_sequence": backbone.sequence,
                "pdb_observed_sequence_length": len(backbone.sequence),
                "pdb_observed_backbone_complete_count": int(backbone.coordinate_mask.sum().item()),
                "pdb_backbone_residue_records": list(backbone.residue_records),
                "alignment_summary": alignment_summary,
                "proteinmpnn_coordinate_mask_true": int(coordinate_mask.sum().item()),
                "proteinmpnn_logits": logits,
                "proteinmpnn_mask": coordinate_mask.detach().cpu().bool().contiguous(),
                "proteinmpnn_alphabet": AMINO_ACIDS_20,
                "proteinmpnn_checkpoint_sha": checkpoint_sha,
                "proteinmpnn_source_sha": source_sha,
                "sequence_sha": sha256_of_string(target_sequence),
                "generated_at": datetime.now().isoformat(),
                "elapsed_seconds": perf_counter() - protein_start,
            }
            torch.save(payload, protein_dir / "proteinmpnn_logits.pt")
            torch.save(payload["proteinmpnn_mask"], protein_dir / "proteinmpnn_mask.pt")
            metadata = {key: value for key, value in payload.items() if not torch.is_tensor(value)}
            (protein_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            manifest_rows.append(
                {
                    "protein_name": job.protein_name,
                    "structure_id": job.structure_id,
                    "pdb_path": work_path_str(job.pdb_path),
                    "chain_id": backbone.chain_id,
                    "sequence_length": len(target_sequence),
                    "coordinate_mask_true": int(coordinate_mask.sum().item()),
                    "sequence_sha": payload["sequence_sha"],
                    "proteinmpnn_checkpoint_sha": checkpoint_sha,
                    "sequence_source": sequence_source,
                }
            )
            print(
                f"[done] {job.protein_name}: {len(target_sequence)} residues "
                f"mask={int(coordinate_mask.sum().item())}/{len(target_sequence)}"
            )
        except Exception as exc:
            failed.append(
                {
                    "protein_name": job.protein_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": perf_counter() - protein_start,
                }
            )
            print(f"[fail] {job.protein_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    write_manifest(output_root / "manifest.csv", manifest_rows)
    summary = {
        "model_family": "ProteinMPNN",
        "feature_style": "canonical_20aa_logits",
        "dataset_csv": work_path_str(resolved_dataset),
        "table_sheet": args.table_sheet,
        "pdb_dir": work_path_str(args.pdb_dir),
        "proteinmpnn_source_module": work_path_str(args.proteinmpnn_source),
        "proteinmpnn_source_sha": source_sha,
        "proteinmpnn_checkpoint": work_path_str(args.proteinmpnn_checkpoint),
        "proteinmpnn_checkpoint_sha": checkpoint_sha,
        "reference_cache_dir": work_path_str(args.reference_cache_dir),
        "sequence_priority": [
            "reference_cache_sequence",
            "workbook_wt_sequence",
            "pdb_atom_record_backbone_fallback",
        ],
        "requested": len(loaded_jobs),
        "matched_pdb": len(jobs),
        "processed": len(manifest_rows),
        "reference_sequence_missing": reference_sequence_missing,
        "pdb_missing": missing_pdbs,
        "failed": failed,
        "total_time_seconds": perf_counter() - start,
        "generated_at": datetime.now().isoformat(),
    }
    (output_root / "proteinmpnn_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not manifest_rows:
        raise SystemExit("No ProteinMPNN cache entries were generated.")


if __name__ == "__main__":
    main()
