#!/usr/bin/env python3
"""Export single-mutant DDG matrices for double-mutation inference.

Run this after the SaProt + ProteinMPNN intrinsic-fusion single model is
finalized and the double-mutation proteins have cached SaProt embeddings plus
ProteinMPNN logits. Pass ``--single-checkpoint`` more than once to average a
seed ensemble into one per-protein DDG matrix. The output directory becomes
``--single-ddg-dir`` for the final double-mutation inference step.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.double_mutation_dataset import (  # noqa: E402
    _candidate_protein_dirs,
    _load_ca_coordinates,
    _resolve_embedding_path,
    load_pdb_derived_wt_sequence,
    load_double_mutation_workbook,
    resolve_work_path,
    torch_load_compatible,
    with_pdb_derived_wt_sequence,
)
from core.pipeline_config import work_path_str  # noqa: E402
from core.proteinmpnn_logits_cache import ProteinMPNNLogitsCache  # noqa: E402
from models.saprot_proteinmpnn_intrinsic_fusion import (  # noqa: E402
    LOCAL_CONTACT_CUTOFF_A,
    LOCAL_CONTACT_DISTANCE_SCALE_A,
    LOCAL_CONTACT_TOP_K,
    SaProtProteinMPNNIntrinsicFusionHead,
)


DATASET_ROOT = "examples/quickstart/double_mutation"
DEFAULT_INPUT_TABLE = f"{DATASET_ROOT}/double_mutations.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export single-mutant DDG matrices for double-mutation proteins.")
    parser.add_argument(
        "--single-checkpoint",
        dest="single_checkpoints",
        action="append",
        required=True,
        help=(
            "Finalized SaProt + ProteinMPNN intrinsic-fusion checkpoint/package. "
            "Pass once per seed; predictions are averaged when multiple are provided."
        ),
    )
    parser.add_argument("--embeddings-dir", required=True, help="SaProt by_protein cache covering the double-mutation proteins.")
    parser.add_argument(
        "--proteinmpnn-cache-dir",
        required=True,
        help="ProteinMPNN logits by_protein cache covering the double-mutation proteins.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to write {pdb_chain}.pt matrices and manifest.csv.")
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--table-sheet", dest="table_sheet", default="refined_sorted")
    parser.add_argument(
        "--baseline-name",
        default="saafec_stair_single_mutation_3seed_ensemble",
        help="Metadata label saved with each exported matrix.",
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


def collect_records(path: str, *, sheet_name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for protein_name, record in load_double_mutation_workbook(path, sheet_name=sheet_name).items():
        output[protein_name] = {
            "record": record,
            "source": work_path_str(resolve_work_path(path)),
        }
    return output


def load_checkpoint_payload(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _model_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("model_config")
    if isinstance(direct, dict):
        return direct
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        nested = value.get("model_config")
        if isinstance(nested, dict):
            return nested
    return {}


def extract_state_dict_and_config(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = load_checkpoint_payload(path, map_location="cpu")
    if isinstance(payload, dict) and "head_state_dict" in payload:
        return payload["head_state_dict"], _model_config_from_payload(payload)
    if isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload, {}
    raise TypeError(
        f"Unsupported checkpoint format for {path}. Expected a state_dict or a package with head_state_dict."
    )


def infer_head_kwargs(
    state_dict: dict[str, torch.Tensor],
    *,
    d_saprot: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    saprot_proj = state_dict.get("saprot_proj.weight")
    if saprot_proj is not None:
        inferred_hidden, inferred_saprot = int(saprot_proj.shape[0]), int(saprot_proj.shape[1])
        if inferred_saprot != d_saprot:
            raise ValueError(
                f"Checkpoint expects SaProt embedding dim {inferred_saprot}, "
                f"but double-mutation cache has dim {d_saprot}."
            )
    else:
        inferred_hidden = int(model_config.get("hidden_dim", 768))

    aa_weight = state_dict.get("aa_embedding.weight")
    aa_embed_dim = int(aa_weight.shape[1]) if aa_weight is not None else int(model_config.get("aa_embed_dim", 64))
    residual_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("residual_blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
    }
    num_residual_blocks = len(residual_indices) if residual_indices else int(model_config.get("residual_blocks", 2))
    return {
        "d_saprot": d_saprot,
        "d_hidden": int(model_config.get("hidden_dim", inferred_hidden)),
        "aa_embed_dim": aa_embed_dim,
        "dropout": 0.0,
        "num_attention_heads": int(model_config.get("attention_heads", 8)),
        "num_residual_blocks": num_residual_blocks,
        "local_contact_top_k": int(model_config.get("local_contact_top_k", LOCAL_CONTACT_TOP_K)),
        "local_contact_cutoff": float(model_config.get("local_contact_cutoff", LOCAL_CONTACT_CUTOFF_A)),
        "local_contact_distance_scale": float(
            model_config.get("local_contact_distance_scale", LOCAL_CONTACT_DISTANCE_SCALE_A)
        ),
    }


def load_intrinsic_head(path: Path, *, d_saprot: int, device: torch.device) -> SaProtProteinMPNNIntrinsicFusionHead:
    state_dict, model_config = extract_state_dict_and_config(path)
    head = SaProtProteinMPNNIntrinsicFusionHead(**infer_head_kwargs(state_dict, d_saprot=d_saprot, model_config=model_config))
    head.load_state_dict(state_dict, strict=True)
    head.to(device)
    head.eval()
    return head


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    embeddings_dir = resolve_work_path(args.embeddings_dir)
    proteinmpnn_cache_dir = resolve_work_path(args.proteinmpnn_cache_dir)
    proteinmpnn_cache = ProteinMPNNLogitsCache(proteinmpnn_cache_dir)
    output_dir = resolve_work_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [resolve_work_path(value) for value in args.single_checkpoints]
    heads: list[SaProtProteinMPNNIntrinsicFusionHead] | None = None
    records = collect_records(args.input_table, sheet_name=args.table_sheet)

    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for protein_name, item in sorted(records.items()):
        record = item["record"]
        protein_dir = None
        emb_path = None
        for candidate_dir in _candidate_protein_dirs(embeddings_dir, record):
            candidate_emb = _resolve_embedding_path(candidate_dir)
            if candidate_emb is not None:
                protein_dir = candidate_dir
                emb_path = candidate_emb
                break
        if protein_dir is None or emb_path is None:
            missing.append(protein_name)
            manifest_rows.append(
                {
                    "protein_name": protein_name,
                    "source": item["source"],
                    "status": "missing_embedding",
                    "embedding_path": "",
                    "output_path": "",
                    "length": len(record.wt_sequence),
                }
            )
            continue

        pdb_sequence = load_pdb_derived_wt_sequence(protein_dir, protein_name)
        record = with_pdb_derived_wt_sequence(
            record,
            pdb_sequence,
            sequence_source=str(protein_dir / "metadata.json"),
        )

        embeddings = torch_load_compatible(emb_path, map_location="cpu", weights_only=True).float()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(record.wt_sequence):
            raise ValueError(
                f"{protein_name}: embedding shape {tuple(embeddings.shape)} does not match "
                f"sequence length {len(record.wt_sequence)}"
            )
        if heads is None:
            d_saprot = int(embeddings.shape[1])
            heads = [
                load_intrinsic_head(checkpoint_path, d_saprot=d_saprot, device=device)
                for checkpoint_path in checkpoint_paths
            ]
        proteinmpnn_logits, proteinmpnn_mask = proteinmpnn_cache.load(
            protein_name,
            expected_length=int(embeddings.shape[0]),
            wt_sequence=record.wt_sequence,
        )
        ca_coordinates = _load_ca_coordinates(protein_dir, len(record.wt_sequence))
        member_ddgs = [
            head(
                embeddings.to(device),
                proteinmpnn_logits.to(device),
                record.wt_sequence,
                ca_coordinates=ca_coordinates.to(device),
                proteinmpnn_mask=proteinmpnn_mask.to(device),
            ).detach().cpu()
            for head in heads
        ]
        stacked_ddg = torch.stack(member_ddgs, dim=0)
        ddg = stacked_ddg.mean(dim=0)
        ddg_std = stacked_ddg.std(dim=0, unbiased=False) if len(member_ddgs) > 1 else torch.zeros_like(ddg)
        output_path = output_dir / f"{protein_name}.pt"
        torch.save(
            {
                "protein_name": protein_name,
                "pdb": record.pdb,
                "chain": record.chain,
                "wt_sequence": record.wt_sequence,
                "ddg": ddg,
                "baseline": args.baseline_name,
                "single_model_family": "saprot_proteinmpnn_intrinsic_fusion",
                "ensemble_n": len(member_ddgs),
                "single_checkpoint_paths": [work_path_str(path) for path in checkpoint_paths],
                "proteinmpnn_cache_dir": work_path_str(proteinmpnn_cache_dir),
                "member_ddg_std": ddg_std,
            },
            output_path,
        )
        manifest_rows.append(
            {
                "protein_name": protein_name,
                "source": item["source"],
                "status": "exported",
                "embedding_path": work_path_str(emb_path),
                "proteinmpnn_cache_dir": work_path_str(proteinmpnn_cache_dir),
                "output_path": work_path_str(output_path),
                "length": len(record.wt_sequence),
                "ensemble_n": len(member_ddgs),
                "single_checkpoint_paths": ";".join(work_path_str(path) for path in checkpoint_paths),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "protein_name",
            "source",
            "status",
            "embedding_path",
            "proteinmpnn_cache_dir",
            "output_path",
            "length",
            "ensemble_n",
            "single_checkpoint_paths",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Exported {sum(row['status'] == 'exported' for row in manifest_rows)} single-mutation prior matrices to {work_path_str(output_dir)}")
    if missing:
        print(f"Missing embeddings for {len(missing)} proteins. See {manifest_path}")


if __name__ == "__main__":
    main()
