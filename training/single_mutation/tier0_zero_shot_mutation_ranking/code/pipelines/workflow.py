#!/usr/bin/env python3
"""Evaluate SaProt 650M PDB Tier 0 fixed-3Di base PLM scores."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from cache.generate_saprot_structure_aware_cache import (  # noqa: E402
    load_saprot_masked_lm,
    resolve_saprot_model_dir,
)
from core.megascale_dataset import MegaScaleDataset  # noqa: E402
from core.pipeline_config import (  # noqa: E402
    DEFAULT_DATABASE_DIR,
    DEFAULT_FOLDSEEK_BIN,
    DEFAULT_SAPROT_MODEL_DIR,
    DEFAULT_STRUCTURE_ROOT,
    ensure_output_root,
    resolve_output_path,
    work_path_str,
)
from core.stability_metrics import evaluate  # noqa: E402
from models.stability_head import AA_TO_INDEX, AMINO_ACIDS_20, NUM_AMINO_ACIDS  # noqa: E402
from training.megascale_test_eval import (  # noqa: E402
    build_mutation_prediction_rows_for_protein,
    print_megascale_test_metrics_payload,
    resolve_test_dataset_specs,
    save_megascale_test_metrics,
    save_prediction_reuse_subset_metrics,
    write_megascale_test_phase_summary,
)


DEFAULT_TEST_XLSX = str(
    DEFAULT_DATABASE_DIR / "01_Single_Mutation/03_Testing/04_Final_Test_Set.xlsx"
)
DEFAULT_TEST_PDB_DIR = str(DEFAULT_STRUCTURE_ROOT / "single_mutation/test")
DEFAULT_OUTPUT_ROOT = "output/tier0_saprot_zero_shot_mutation_ranking"
PHASE_NAME = "tier0_saprot_zero_shot_mutation_ranking"
CANONICAL_AA = list(AMINO_ACIDS_20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description=(
            "Run Tier 0 SaProt 650M PDB zero-shot mutation ranking. "
            "No supervised head and no ProteinMPNN features are used."
        )
    )
    parser.add_argument("--test-xlsx", action="append", default=None)
    parser.add_argument("--test-pdb-dir", action="append", default=None)
    parser.add_argument(
        "--test-structure-set",
        default="colabfold",
        choices=[
            "colabfold",
            "colabfold_rank45_subset",
            "modelled_rank45",
            "modeled_rank45",
        ],
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--saprot-model-dir", default=str(DEFAULT_SAPROT_MODEL_DIR))
    parser.add_argument("--foldseek-bin", default=str(DEFAULT_FOLDSEEK_BIN))
    parser.add_argument("--model-name", default="SaProt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--mask-batch-size",
        type=int,
        default=8,
        help="Number of masked positions scored per SaProt forward pass.",
    )
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument(
        "--score-all-residues",
        action="store_true",
        help=(
            "Score all residues in each protein. Default scores only positions "
            "with measured mutations because unmeasured positions do not enter metrics."
        ),
    )
    args = parser.parse_args()
    if args.test_xlsx is None:
        args.test_xlsx = [DEFAULT_TEST_XLSX]
    if args.test_pdb_dir is None:
        args.test_pdb_dir = [DEFAULT_TEST_PDB_DIR]
    if args.mask_batch_size < 1:
        parser.error("--mask-batch-size must be >= 1.")
    return args


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def run_stage(cmd: list[str], stage_name: str) -> None:
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=WORK_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_saprot_cache_cmd(
    *,
    pdb_dir: str,
    dataset_xlsx: str,
    xlsx_sheet: str,
    embeddings_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS_ROOT / "cache" / "generate_saprot_structure_aware_cache.py"),
        "--dataset-csv",
        dataset_xlsx,
        "--xlsx-sheet",
        xlsx_sheet,
        "--pdb-dir",
        pdb_dir,
        "--output-dir",
        str(embeddings_root),
        "--saprot-model-dir",
        args.saprot_model_dir,
        "--foldseek-bin",
        args.foldseek_bin,
        "--model-name",
        args.model_name,
        "--device",
        args.device,
    ]


def load_torch_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cached_input_tensors(
    *,
    protein_dir: Path,
    tokenizer: Any,
    combined_sequence: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids_path = protein_dir / "inputs" / "saprot" / "input_ids.pt"
    attention_mask_path = protein_dir / "inputs" / "saprot" / "attention_mask.pt"
    if input_ids_path.is_file() and attention_mask_path.is_file():
        input_ids = load_torch_payload(input_ids_path).long()
        attention_mask = load_torch_payload(attention_mask_path).long()
    else:
        encoded = tokenizer(
            combined_sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
    return input_ids.to(device), attention_mask.to(device)


def token_id_for_fixed_3di(tokenizer: Any, aa: str, struct_token: str) -> int:
    vocab = tokenizer.get_vocab()
    candidates = [
        f"{aa}{struct_token.lower()}",
        f"{aa}{struct_token}",
        f"{aa}{struct_token.upper()}",
    ]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in candidates:
        if token not in vocab:
            continue
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if unk_id is None or token_id != int(unk_id):
            return token_id
    raise KeyError(
        f"Could not map amino acid {aa!r} with fixed 3Di token {struct_token!r} "
        "to a SaProt tokenizer id."
    )


def fixed_3di_token_ids_for_position(tokenizer: Any, struct_token: str) -> torch.Tensor:
    return torch.tensor(
        [token_id_for_fixed_3di(tokenizer, aa, struct_token) for aa in CANONICAL_AA],
        dtype=torch.long,
    )


def load_cache_metadata(embeddings_root: Path, protein_name: str) -> tuple[Path, dict[str, Any]]:
    protein_dir = embeddings_root / protein_name
    metadata_path = protein_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing SaProt cache metadata for {protein_name}: {metadata_path}")
    return protein_dir, json.loads(metadata_path.read_text(encoding="utf-8"))


@torch.no_grad()
def score_tier0_sample(
    *,
    sample: Any,
    embeddings_root: Path,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    mask_batch_size: int,
    score_all_residues: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return an ``L x 20`` DDG-like matrix from fixed-3Di masked-marginal scores."""
    protein_dir, metadata = load_cache_metadata(embeddings_root, sample.protein_name)
    wt_sequence = str(metadata.get("sequence") or "")
    foldseek_3di = str(metadata.get("foldseek_3di_sequence") or "")
    combined_sequence = str(metadata.get("structure_aware_sequence") or "")
    if not wt_sequence or not foldseek_3di or not combined_sequence:
        raise ValueError(f"SaProt cache metadata is incomplete for {sample.protein_name}.")
    if wt_sequence != sample.wt_sequence:
        raise ValueError(
            f"WT sequence mismatch for {sample.protein_name}: cache L={len(wt_sequence)} "
            f"dataset L={len(sample.wt_sequence)}."
        )
    if len(foldseek_3di) != len(wt_sequence):
        raise ValueError(
            f"Foldseek 3Di length {len(foldseek_3di)} does not match sequence length "
            f"{len(wt_sequence)} for {sample.protein_name}."
        )

    input_ids, attention_mask = load_cached_input_tensors(
        protein_dir=protein_dir,
        tokenizer=tokenizer,
        combined_sequence=combined_sequence,
        device=device,
    )
    expected_tokens = len(wt_sequence) + 2
    if input_ids.ndim != 2 or input_ids.shape[1] != expected_tokens:
        raise ValueError(
            f"SaProt token count mismatch for {sample.protein_name}: "
            f"got {tuple(input_ids.shape)}, expected (1, {expected_tokens})."
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("SaProt tokenizer does not define a mask token id.")

    if score_all_residues:
        positions = list(range(len(wt_sequence)))
    else:
        positions = torch.nonzero(sample.mask.sum(dim=1) > 0, as_tuple=False).flatten().tolist()

    ddg_like = torch.zeros((len(wt_sequence), NUM_AMINO_ACIDS), dtype=torch.float32)
    log_odds = torch.zeros_like(ddg_like)
    wt_log_probs = torch.full((len(wt_sequence),), float("nan"), dtype=torch.float32)
    scored_positions: list[int] = []
    token_cache: dict[str, torch.Tensor] = {}

    for start in range(0, len(positions), mask_batch_size):
        batch_positions = [int(pos) for pos in positions[start : start + mask_batch_size]]
        if not batch_positions:
            continue
        batch_input_ids = input_ids.repeat(len(batch_positions), 1)
        batch_attention = attention_mask.repeat(len(batch_positions), 1)
        for row_idx, pos_idx in enumerate(batch_positions):
            batch_input_ids[row_idx, pos_idx + 1] = int(tokenizer.mask_token_id)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention)
        logits = outputs.logits
        for row_idx, pos_idx in enumerate(batch_positions):
            struct_token = foldseek_3di[pos_idx]
            if struct_token not in token_cache:
                token_cache[struct_token] = fixed_3di_token_ids_for_position(
                    tokenizer,
                    struct_token,
                ).to(device)
            aa_token_ids = token_cache[struct_token]
            log_probs = torch.log_softmax(logits[row_idx, pos_idx + 1], dim=-1)
            aa_log_probs = log_probs.index_select(0, aa_token_ids).detach().cpu()
            wt_idx = AA_TO_INDEX[wt_sequence[pos_idx]]
            position_log_odds = aa_log_probs - aa_log_probs[wt_idx]
            log_odds[pos_idx] = position_log_odds
            ddg_like[pos_idx] = -position_log_odds
            wt_log_probs[pos_idx] = aa_log_probs[wt_idx]
            scored_positions.append(pos_idx + 1)

    score_metadata = {
        "protein_name": sample.protein_name,
        "sequence_length": len(wt_sequence),
        "scored_position_count": len(scored_positions),
        "scored_positions_1based": scored_positions,
        "score_definition": "base_ddg_like = -[logP(mutant AA with fixed 3Di_i) - logP(WT AA with fixed 3Di_i)]",
        "base_log_odds_matrix": log_odds,
        "wt_log_prob_by_position": wt_log_probs,
    }
    return ddg_like, score_metadata


def save_tier0_score_artifacts(
    *,
    output_dir: Path,
    score_metadata_by_protein: list[dict[str, Any]],
) -> None:
    matrix_dir = output_dir / "tier0_score_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for metadata in score_metadata_by_protein:
        protein_name = str(metadata["protein_name"])
        torch.save(
            metadata["base_log_odds_matrix"],
            matrix_dir / f"{protein_name}.base_log_odds.pt",
        )
        torch.save(
            metadata["wt_log_prob_by_position"],
            matrix_dir / f"{protein_name}.wt_log_prob.pt",
        )
        summary_rows.append(
            {
                "protein_name": protein_name,
                "sequence_length": metadata["sequence_length"],
                "scored_position_count": metadata["scored_position_count"],
                "score_definition": metadata["score_definition"],
            }
        )
    with (output_dir / "tier0_score_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["protein_name", "sequence_length", "scored_position_count", "score_definition"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def evaluate_tier0_dataset(
    *,
    spec: Any,
    embeddings_root: Path,
    output_dir: Path,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    model_dir: Path,
    mask_batch_size: int,
    score_all_residues: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = MegaScaleDataset(
        mutations_table=spec.table,
        embeddings_dir=embeddings_root,
        split="test",
        splits_csv=None,
        workbook_is_split=True,
        xlsx_sheet_name=spec.sheet_name,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Held-out Tier 0 dataset is empty after cache matching: {spec.name}")

    ddg_pred_list: list[torch.Tensor] = []
    ddg_true_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    protein_names: list[str] = []
    source_databases: list[str] = []
    mutation_prediction_rows: list[dict[str, Any]] = []
    score_metadata_by_protein: list[dict[str, Any]] = []
    per_protein_mse: list[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        ddg_pred, score_metadata = score_tier0_sample(
            sample=sample,
            embeddings_root=embeddings_root,
            tokenizer=tokenizer,
            model=model,
            device=device,
            mask_batch_size=mask_batch_size,
            score_all_residues=score_all_residues,
        )
        target = sample.target.float()
        mask = sample.mask.float()
        active = mask > 0
        if active.sum() == 0:
            continue
        per_protein_mse.append(float(((ddg_pred - target).pow(2) * mask).sum().item() / mask.sum().item()))
        ddg_pred_list.append(ddg_pred)
        ddg_true_list.append(target)
        mask_list.append(mask)
        protein_names.append(sample.protein_name)
        source_databases.append(sample.source_database)
        score_metadata_by_protein.append(score_metadata)
        mutation_prediction_rows.extend(
            build_mutation_prediction_rows_for_protein(
                ddg_pred=ddg_pred,
                target=target,
                mask=mask,
                wt_sequence=sample.wt_sequence,
                protein_name=sample.protein_name,
                mutation_resolution=sample.mutation_resolution,
            )
        )

    if not ddg_pred_list:
        raise RuntimeError(f"No resolved Tier 0 mutation predictions for dataset {spec.name}.")

    test_result = evaluate(
        ddg_pred_list,
        ddg_true_list,
        mask_list,
        protein_names,
        source_databases,
    )
    test_loss = sum(per_protein_mse) / len(per_protein_mse)
    payload = save_megascale_test_metrics(
        output_dir=output_dir,
        phase=0,
        phase_name=PHASE_NAME,
        epoch=0,
        checkpoint_path=model_dir,
        test_table=spec.table,
        test_loss=test_loss,
        test_result=test_result,
        test_name=spec.name,
        test_xlsx_sheet=spec.sheet_name,
        test_structure_set=spec.structure_set,
        mutation_prediction_rows=mutation_prediction_rows,
    )
    payload["tier"] = 0
    payload["tier_name"] = "Base PLM Score"
    payload["score_definition"] = (
        "base_ddg_like(i,a) = -[logP_SaProt(AA=a with fixed 3Di_i | masked context) "
        "- logP_SaProt(AA=WT_i with fixed 3Di_i | masked context)]"
    )
    payload["uses_supervised_head"] = False
    payload["uses_proteinmpnn"] = False
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_tier0_score_artifacts(
        output_dir=output_dir,
        score_metadata_by_protein=score_metadata_by_protein,
    )
    return payload, mutation_prediction_rows


def main() -> None:
    args = parse_args()
    output_root = ensure_output_root(args.output_root)
    embeddings_root = output_root / "embeddings" / "by_protein"
    model_dir = resolve_saprot_model_dir(args.saprot_model_dir)
    test_specs = resolve_test_dataset_specs(
        args.test_xlsx,
        args.test_pdb_dir,
        structure_set=args.test_structure_set,
    )
    real_specs = [spec for spec in test_specs if spec.prediction_source_name is None]
    if not real_specs:
        raise RuntimeError("No concrete Tier 0 test datasets were resolved.")

    if not args.skip_cache:
        cache_root = output_root / "embeddings"
        for spec in real_specs:
            if spec.pdb_dir is None:
                raise SystemExit(f"Missing PDB directory for test dataset {spec.name}.")
            run_stage(
                build_saprot_cache_cmd(
                    pdb_dir=spec.pdb_dir,
                    dataset_xlsx=spec.table,
                    xlsx_sheet=spec.sheet_name,
                    embeddings_root=cache_root,
                    args=args,
                ),
                f"Cache SaProt 650M PDB structure-aware inputs ({spec.name})",
            )
    else:
        print("\nSkipping SaProt cache generation and reusing existing embeddings.")

    if not embeddings_root.is_dir():
        raise FileNotFoundError(f"SaProt embeddings cache missing: {embeddings_root}")

    from transformers import EsmForMaskedLM, EsmTokenizer  # type: ignore

    device = choose_device(args.device)
    tokenizer = EsmTokenizer.from_pretrained(str(model_dir))
    model, model_load_mode = load_saprot_masked_lm(model_dir, EsmForMaskedLM)
    model = model.to(device).eval()
    print(
        f"Loaded SaProt Tier 0 model from {model_dir} on {device} "
        f"({model_load_mode}); mask_batch_size={args.mask_batch_size}"
    )

    invocation_summary = {
        "workflow": "saprot_650m_pdb_tier0_base_plm_score",
        "tier": 0,
        "phase_name": PHASE_NAME,
        "model_dir": work_path_str(model_dir),
        "model_load_mode": model_load_mode,
        "device": str(device),
        "mask_batch_size": args.mask_batch_size,
        "score_all_residues": args.score_all_residues,
        "test_specs": [spec.__dict__ for spec in test_specs],
        "output_root": work_path_str(output_root),
    }
    (output_root / "tier0_invocation_summary.json").write_text(
        json.dumps(invocation_summary, indent=2),
        encoding="utf-8",
    )

    source_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    metric_paths: list[Path] = []
    payloads: list[tuple[dict[str, Any], Path]] = []
    for spec in test_specs:
        test_output_dir = output_root / "test_metrics" / spec.name / "tier0_base_plm"
        if spec.prediction_source_name is not None:
            source_rows = source_rows_by_dataset.get(spec.prediction_source_name)
            if source_rows is None:
                raise RuntimeError(
                    f"Prediction-reuse dataset {spec.name} requested source "
                    f"{spec.prediction_source_name}, but that source has not been evaluated."
                )
            payload = save_prediction_reuse_subset_metrics(
                output_dir=test_output_dir,
                phase=0,
                phase_name=PHASE_NAME,
                epoch=0,
                checkpoint_path=model_dir,
                subset_spec=spec,
                source_rows=source_rows,
            )
            payload["tier"] = 0
            payload["tier_name"] = "Base PLM Score"
            payload["uses_supervised_head"] = False
            payload["uses_proteinmpnn"] = False
            (test_output_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            mutation_rows = source_rows
        else:
            payload, mutation_rows = evaluate_tier0_dataset(
                spec=spec,
                embeddings_root=embeddings_root,
                output_dir=test_output_dir,
                tokenizer=tokenizer,
                model=model,
                device=device,
                model_dir=model_dir,
                mask_batch_size=args.mask_batch_size,
                score_all_residues=args.score_all_residues,
            )
            source_rows_by_dataset[spec.name] = mutation_rows
        metric_paths.append(test_output_dir / "metrics.json")
        payloads.append((payload, test_output_dir))

    write_megascale_test_phase_summary(
        output_dir=output_root / "test_metrics",
        metric_paths=metric_paths,
        summary_stem="tier0_test_metrics",
    )
    for payload, test_output_dir in payloads:
        print_megascale_test_metrics_payload(payload, output_dir=test_output_dir)
    print(f"\nWrote Tier 0 metrics for {len(payloads)} datasets to {output_root / 'test_metrics'}")


if __name__ == "__main__":
    main()
