#!/usr/bin/env python3
"""
generate_saprot_structure_aware_cache.py
----------------------------------------
Cache per-residue SaProt embeddings for the single-phase SAAFEC workflow.

SaProt uses a structure-aware vocabulary: each residue is represented by an
amino-acid token fused with a Foldseek 3Di structure token. This script converts
each PDB chain into that structure-aware sequence, runs a Hugging Face SaProt
masked-LM checkpoint, and saves one ``per_residue_embeddings.pt`` tensor per
protein in the same by_protein layout consumed by ``MegaScaleDataset``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.megascale_dataset import (  # noqa: E402
    load_protein_chain_map,
    load_protein_names_csv,
    load_protein_structure_map,
    resolve_dataset_csv_path,
)
from core.pipeline_config import (  # noqa: E402
    WORK_DIR,
    ensure_output_root,
    list_pdb_files,
    resolve_model_path,
    resolve_work_path,
    work_path_str,
)
from utils.output_error_logging import OutputErrorLogger, infer_output_base  # noqa: E402
from utils.pipeline_io import write_per_residue_csv  # noqa: E402


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
    "SEC": "U",
    "PYL": "O",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SaProt structure-aware per-residue embedding caches."
    )
    parser.add_argument("--dataset-csv", required=True, help="Workbook/CSV listing proteins to cache.")
    parser.add_argument("--xlsx-sheet", default="refined_sorted_clean")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing WT PDB files.")
    parser.add_argument("--output-dir", default="output/saprot_phase1/embeddings")
    parser.add_argument(
        "--saprot-model-dir",
        default="SaProt",
        help=(
            "Hugging Face SaProt checkpoint directory. Relative paths are resolved "
            "under the work dir first, then model directories."
        ),
    )
    parser.add_argument(
        "--foldseek-bin",
        default="foldseek",
        help="Foldseek executable used for structureto3didescriptor.",
    )
    parser.add_argument("--model-name", default="SaProt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_saprot_model_dir(raw_path: str | Path) -> Path:
    candidates = [
        resolve_work_path(raw_path),
        resolve_model_path(raw_path),
        Path(raw_path).expanduser().resolve(),
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
        if candidate.is_dir():
            config_paths = sorted(candidate.glob("**/config.json"))
            if len(config_paths) == 1:
                return config_paths[0].parent
            preferred_names = [
                "westlake-repl_SaProt_650M_AF2",
                "SaProt_650M_AF2",
                "SaProt_650M_PDB",
                "SaProt_35M_AF2",
                "SaProt_1.3B_AFDB_OMG_NCBI",
                "SaProt_1.3B_AF2",
            ]
            for name in preferred_names:
                preferred_paths = [
                    path.parent
                    for path in config_paths
                    if path.parent.name == name
                ]
                if preferred_paths:
                    return preferred_paths[0]
            if len(config_paths) > 1:
                found = "\n".join(f"  - {path.parent}" for path in config_paths)
                raise FileNotFoundError(
                    "Multiple SaProt Hugging Face checkpoints found. Pass "
                    f"--saprot-model-dir explicitly.\n{found}"
                )
    raise FileNotFoundError(
        f"Could not find a Hugging Face SaProt checkpoint at or under: {raw_path}"
    )


def resolve_foldseek_bin(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [
        resolve_work_path(path),
        Path(raw_path).expanduser().resolve(),
    ]
    which = shutil.which(str(raw_path))
    if which:
        candidates.append(Path(which).resolve())
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"Foldseek executable not found or not executable: {raw_path}"
    )


def infer_chain_id_from_stem(stem: str) -> str | None:
    parts = stem.split("_")
    if len(parts) >= 2 and len(parts[-1]) == 1:
        return parts[-1]
    return None


def parse_pdb_residue_mapping(pdb_path: Path, chain_id: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    observed_index = 0
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_chain = line[21].strip() or " "
            if chain_id is not None and atom_chain != chain_id:
                continue
            resname = line[17:20].strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            ins_code = line[26].strip()
            key = (atom_chain, resseq, ins_code)
            if key not in record_by_key:
                observed_index += 1
                residue_id = f"{atom_chain}:{resseq}{ins_code}"
                record = {
                    "model_index": observed_index,
                    "aa": aa,
                    "chain_id": atom_chain,
                    "observed": True,
                    "observed_index": observed_index,
                    "pdb_resseq": resseq,
                    "pdb_ins_code": ins_code,
                    "pdb_residue": residue_id,
                    "pdb_residue_id": residue_id,
                    "has_ca": False,
                    "ca_x": None,
                    "ca_y": None,
                    "ca_z": None,
                }
                record_by_key[key] = record
                records.append(record)
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                try:
                    ca_x = float(line[30:38])
                    ca_y = float(line[38:46])
                    ca_z = float(line[46:54])
                except ValueError:
                    continue
                record_by_key[key].update(
                    {
                        "has_ca": True,
                        "ca_x": ca_x,
                        "ca_y": ca_y,
                        "ca_z": ca_z,
                    }
                )
    return records


def run_foldseek_structure_descriptor(
    *,
    foldseek_bin: Path,
    pdb_path: Path,
    chain_id: str | None,
) -> tuple[str, str, str]:
    with tempfile.TemporaryDirectory(prefix="saprot_foldseek_") as tmp:
        out_tsv = Path(tmp) / "structure_3di.tsv"
        cmd = [
            str(foldseek_bin),
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(pdb_path),
            str(out_tsv),
        ]
        subprocess.run(cmd, check=True)
        if not out_tsv.is_file():
            raise FileNotFoundError(f"Foldseek did not produce expected TSV: {out_tsv}")
        parsed: dict[str, tuple[str, str, str]] = {}
        pdb_name = pdb_path.name
        with out_tsv.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                desc, aa_seq, foldseek_seq = fields[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(pdb_name, "").split("_")[-1]
                combined_seq = "".join(
                    aa + struct.lower() for aa, struct in zip(aa_seq, foldseek_seq)
                )
                parsed[chain] = (aa_seq, foldseek_seq, combined_seq)
        if chain_id is not None and chain_id in parsed:
            return parsed[chain_id]
        if chain_id is None and len(parsed) == 1:
            return next(iter(parsed.values()))
        if parsed:
            available = ", ".join(sorted(parsed))
            raise ValueError(
                f"Could not resolve chain {chain_id!r} from Foldseek output for "
                f"{pdb_path.name}. Available chains: {available}"
            )
        raise ValueError(f"Foldseek produced no chain records for {pdb_path}")


def write_residue_mapping_csv(path: Path, residue_mapping: list[dict[str, Any]]) -> None:
    fields = [
        "model_index",
        "aa",
        "chain_id",
        "observed",
        "observed_index",
        "pdb_resseq",
        "pdb_ins_code",
        "pdb_residue",
        "pdb_residue_id",
        "has_ca",
        "ca_x",
        "ca_y",
        "ca_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in residue_mapping:
            writer.writerow({field: record.get(field) for field in fields})


def build_ca_coordinate_tensor(residue_mapping: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C-alpha coordinates and a validity mask aligned to model residues."""
    coords = torch.full((len(residue_mapping), 3), float("nan"), dtype=torch.float32)
    mask = torch.zeros((len(residue_mapping),), dtype=torch.bool)
    for idx, record in enumerate(residue_mapping):
        if not record.get("has_ca"):
            continue
        try:
            coords[idx] = torch.tensor(
                [
                    float(record["ca_x"]),
                    float(record["ca_y"]),
                    float(record["ca_z"]),
                ],
                dtype=torch.float32,
            )
        except (TypeError, ValueError, KeyError):
            continue
        mask[idx] = True
    return coords, mask


def load_saprot_masked_lm(model_dir: Path, model_cls: Any) -> tuple[torch.nn.Module, str]:
    """
    Load a local SaProt Hugging Face checkpoint.

    Recent Transformers versions block `.bin` checkpoint loading with torch<2.6
    through `from_pretrained` because of CVE-2025-32434. These HPC SaProt
    weights are a trusted local lab asset, so fall back to a direct state-dict
    load only for that specific safety gate and only from a resolved local path.
    """
    try:
        return model_cls.from_pretrained(str(model_dir)), "transformers_from_pretrained"
    except ValueError as exc:
        message = str(exc)
        checkpoint_path = model_dir / "pytorch_model.bin"
        if "torch.load" not in message or not checkpoint_path.is_file():
            raise

        from transformers import AutoConfig  # type: ignore

        config = AutoConfig.from_pretrained(str(model_dir))
        model = model_cls(config)
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
            state_dict = state_dict["state_dict"]
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("esm.contact_head.")
        ]
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.endswith("position_ids")
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Manual SaProt checkpoint load had incompatible keys: "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )
        return model, "manual_trusted_local_pytorch_bin"


def main() -> None:
    args = parse_args()
    output_root = ensure_output_root(args.output_dir)
    error_logger = OutputErrorLogger(
        "cache_saprot_embeddings",
        infer_output_base(output_root, WORK_DIR),
    )

    try:
        from transformers import EsmForMaskedLM, EsmTokenizer  # type: ignore

        saprot_model_dir = resolve_saprot_model_dir(args.saprot_model_dir)
        foldseek_bin = resolve_foldseek_bin(args.foldseek_bin)
        pdb_dir = resolve_work_path(args.pdb_dir)
        if not pdb_dir.is_dir():
            raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

        resolved_dataset_csv = resolve_dataset_csv_path(args.dataset_csv)
        protein_names = load_protein_names_csv(
            resolved_dataset_csv,
            xlsx_sheet_name=args.xlsx_sheet,
        )
        structure_map = load_protein_structure_map(
            resolved_dataset_csv,
            xlsx_sheet_name=args.xlsx_sheet,
        )
        chain_map = load_protein_chain_map(
            resolved_dataset_csv,
            xlsx_sheet_name=args.xlsx_sheet,
        )
        available_pdbs = {path.stem: path for path in list_pdb_files(pdb_dir)}
        available_pdbs_lower = {path.stem.lower(): path for path in available_pdbs.values()}
        protein_jobs: list[dict[str, Any]] = []
        missing_pdb_proteins: list[str] = []
        for protein_name in protein_names:
            structure_id = structure_map.get(protein_name, protein_name)
            pdb_path = available_pdbs.get(structure_id) or available_pdbs_lower.get(structure_id.lower())
            if pdb_path is None:
                missing_pdb_proteins.append(protein_name)
                continue
            protein_jobs.append(
                {
                    "protein_name": protein_name,
                    "structure_id": structure_id,
                    "chain_id": (
                        chain_map.get(protein_name)
                        or infer_chain_id_from_stem(protein_name)
                        or infer_chain_id_from_stem(structure_id)
                    ),
                    "pdb_path": pdb_path,
                }
            )
        if not protein_jobs:
            raise FileNotFoundError(
                f"No matching PDB files found in {pdb_dir} for {resolved_dataset_csv}"
            )

        device = choose_device(args.device)
        tokenizer = EsmTokenizer.from_pretrained(str(saprot_model_dir))
        model, model_load_mode = load_saprot_masked_lm(saprot_model_dir, EsmForMaskedLM)
        model = model.to(device).eval()
        model_dtype = next(model.parameters()).dtype
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
        max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)

        by_protein_root = output_root / "by_protein"
        aggregate_embedding_dir = output_root / "aggregates" / "per_residue_embeddings"
        aggregate_tensor_dir = aggregate_embedding_dir / "tensors"
        aggregate_csv_dir = aggregate_embedding_dir / "per_protein_csv"
        by_protein_root.mkdir(parents=True, exist_ok=True)
        aggregate_tensor_dir.mkdir(parents=True, exist_ok=True)
        aggregate_csv_dir.mkdir(parents=True, exist_ok=True)

        invocation_summary: dict[str, Any] = {
            "model_name": args.model_name,
            "model_dir": work_path_str(saprot_model_dir),
            "foldseek_bin": work_path_str(foldseek_bin),
            "device": str(device),
            "model_dtype": str(model_dtype),
            "model_load_mode": model_load_mode,
            "hidden_size": hidden_size,
            "max_position_embeddings": max_positions,
            "timestamp": datetime.now().isoformat(),
            "pdb_dir": work_path_str(pdb_dir),
            "dataset_csv": work_path_str(resolved_dataset_csv),
            "xlsx_sheet": args.xlsx_sheet,
            "requested_proteins": protein_names,
            "missing_pdb_proteins": missing_pdb_proteins,
            "processed": [],
            "failed": [],
            "protein_processing_stats": [],
        }

        for job in protein_jobs:
            protein_start = perf_counter()
            protein_name = job["protein_name"]
            structure_id = job["structure_id"]
            chain_id = job["chain_id"]
            pdb_path = job["pdb_path"]
            run_dir = by_protein_root / protein_name
            if run_dir.exists():
                shutil.rmtree(run_dir)
            emb_dir = run_dir / "outputs" / "embeddings"
            inputs_dir = run_dir / "inputs" / "saprot"
            numeric_dir = run_dir / "inputs" / "numeric"
            emb_dir.mkdir(parents=True, exist_ok=True)
            inputs_dir.mkdir(parents=True, exist_ok=True)
            numeric_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run ] {pdb_path.name}")
            try:
                aa_seq, foldseek_seq, combined_seq = run_foldseek_structure_descriptor(
                    foldseek_bin=foldseek_bin,
                    pdb_path=pdb_path,
                    chain_id=chain_id,
                )
                residue_mapping = parse_pdb_residue_mapping(pdb_path, chain_id)
                if len(residue_mapping) == len(aa_seq):
                    mapping_sequence = "".join(str(record["aa"]) for record in residue_mapping)
                    if mapping_sequence != aa_seq:
                        raise ValueError(
                            f"PDB residue mapping sequence differs from Foldseek AA sequence "
                            f"for {protein_name}."
                        )
                else:
                    residue_mapping = [
                        {
                            "model_index": idx,
                            "aa": aa,
                            "chain_id": chain_id,
                            "observed": True,
                            "observed_index": idx,
                            "pdb_resseq": idx,
                            "pdb_ins_code": "",
                            "pdb_residue": f"{chain_id or ''}:{idx}",
                            "pdb_residue_id": f"{chain_id or ''}:{idx}",
                            "has_ca": False,
                            "ca_x": None,
                            "ca_y": None,
                            "ca_z": None,
                        }
                        for idx, aa in enumerate(aa_seq, start=1)
                    ]
                ca_coordinates, ca_coordinate_mask = build_ca_coordinate_tensor(residue_mapping)

                if max_positions and len(aa_seq) + 2 > max_positions:
                    raise ValueError(
                        f"{protein_name} has {len(aa_seq)} residues, exceeding SaProt "
                        f"max_position_embeddings-2 ({max_positions - 2})."
                    )

                encoded = tokenizer(
                    combined_seq,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                token_count = int(encoded["input_ids"].shape[1])
                if token_count != len(aa_seq) + 2:
                    raise ValueError(
                        f"SaProt tokenizer produced {token_count} tokens for {protein_name}; "
                        f"expected sequence length + BOS/EOS = {len(aa_seq) + 2}."
                    )
                with torch.no_grad():
                    output = model(**encoded, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                residue_embeddings = hidden[:, 1:-1, :].squeeze(0).contiguous()
                if residue_embeddings.shape[0] != len(aa_seq):
                    raise ValueError(
                        f"Residue embedding length mismatch for {protein_name}: "
                        f"{residue_embeddings.shape[0]} vs {len(aa_seq)}"
                    )
                mean_embedding = residue_embeddings.mean(dim=0)

                torch.save(residue_embeddings.detach().cpu(), emb_dir / "per_residue_embeddings.pt")
                torch.save(mean_embedding.detach().cpu(), emb_dir / "mean_embedding.pt")
                torch.save(hidden.detach().cpu(), emb_dir / "embeddings_with_special_tokens.pt")
                torch.save(output.logits.detach().cpu(), run_dir / "outputs" / "embeddings" / "saprot_logits.pt")
                torch.save(encoded["input_ids"].detach().cpu(), inputs_dir / "input_ids.pt")
                torch.save(encoded["attention_mask"].detach().cpu(), inputs_dir / "attention_mask.pt")
                torch.save(ca_coordinates, numeric_dir / "ca_coordinates.pt")
                torch.save(ca_coordinate_mask, numeric_dir / "ca_coordinate_mask.pt")
                torch.save(residue_embeddings.detach().cpu(), aggregate_tensor_dir / f"{protein_name}.pt")
                write_per_residue_csv(residue_embeddings, emb_dir / "per_residue_embeddings.csv")
                write_per_residue_csv(residue_embeddings, aggregate_csv_dir / f"{protein_name}.csv")

                residue_mapping_path = run_dir / "residue_mapping.csv"
                write_residue_mapping_csv(residue_mapping_path, residue_mapping)
                metadata = {
                    "protein_name": protein_name,
                    "structure_id": structure_id,
                    "pdb_chain": protein_name,
                    "model_name": args.model_name,
                    "model_family": "SaProt",
                    "model_dir": work_path_str(saprot_model_dir),
                    "foldseek_bin": work_path_str(foldseek_bin),
                    "device": str(device),
                    "pdb_path": work_path_str(pdb_path),
                    "chain_id": chain_id,
                    "sequence": aa_seq,
                    "sequence_source": "foldseek_structureto3didescriptor",
                    "sequence_length": len(aa_seq),
                    "foldseek_3di_sequence": foldseek_seq,
                    "structure_aware_sequence": combined_seq,
                    "residue_mapping": residue_mapping,
                    "residue_mapping_csv": work_path_str(residue_mapping_path),
                    "ca_coordinates_path": work_path_str(numeric_dir / "ca_coordinates.pt"),
                    "ca_coordinate_mask_path": work_path_str(numeric_dir / "ca_coordinate_mask.pt"),
                    "ca_coordinate_count": int(ca_coordinate_mask.sum().item()),
                    "input_ids_shape": list(encoded["input_ids"].shape),
                    "attention_mask_shape": list(encoded["attention_mask"].shape),
                    "raw_embeddings_shape": list(hidden.shape),
                    "cached_embeddings_shape": list(residue_embeddings.shape),
                    "embedding_dim": int(residue_embeddings.shape[1]),
                    "embeddings_norm": "SaProt final hidden state",
                    "cached_embeddings_exclude_bos_eos": True,
                    "tracks_provided": ["amino_acid_sequence", "foldseek_3di_structure_tokens"],
                    "output_mode": "saprot_embedding_cache_only",
                    "generated_at": datetime.now().isoformat(),
                }
                (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                elapsed = perf_counter() - protein_start
                invocation_summary["processed"].append(protein_name)
                invocation_summary["protein_processing_stats"].append(
                    {
                        "pdb_name": protein_name,
                        "status": "processed",
                        "num_residues": len(aa_seq),
                        "embedding_dim": int(residue_embeddings.shape[1]),
                        "total_time_seconds": elapsed,
                        "error_message": "",
                    }
                )
                print(f"[done] {protein_name} -> {run_dir}")
            except Exception as exc:
                elapsed = perf_counter() - protein_start
                invocation_summary["failed"].append(
                    {"protein_name": protein_name, "pdb": pdb_path.name, "error": str(exc)}
                )
                invocation_summary["protein_processing_stats"].append(
                    {
                        "pdb_name": protein_name,
                        "status": "not_processed",
                        "num_residues": "",
                        "embedding_dim": "",
                        "total_time_seconds": elapsed,
                        "error_message": str(exc),
                    }
                )
                error_logger.exception(
                    "SAPROT_PDB_PROCESSING_ERROR",
                    f"Failed while processing {protein_name} from {pdb_path.name}.",
                    exc,
                    context={"protein_name": protein_name, "pdb_path": work_path_str(pdb_path)},
                )

        summary_path = output_root / "saprot_cache_invocation_summary.json"
        summary_path.write_text(json.dumps(invocation_summary, indent=2), encoding="utf-8")
        if invocation_summary["failed"]:
            error_logger.write_run_status(
                "partial",
                summary={
                    "processed_count": len(invocation_summary["processed"]),
                    "failed_count": len(invocation_summary["failed"]),
                    "summary_path": work_path_str(summary_path),
                },
            )
        else:
            error_logger.write_run_status(
                "completed",
                summary={
                    "processed_count": len(invocation_summary["processed"]),
                    "failed_count": 0,
                    "summary_path": work_path_str(summary_path),
                },
            )
    except Exception as exc:
        error_logger.exception(
            "SAPROT_CACHE_FATAL",
            "SaProt cache generation failed before/after the per-protein loop.",
            exc,
        )
        error_logger.write_run_status("failed", summary={"failure_reason": str(exc)})
        raise


if __name__ == "__main__":
    main()
