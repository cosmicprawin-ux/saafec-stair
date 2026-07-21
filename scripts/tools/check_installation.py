#!/usr/bin/env python3
"""Validate the public source tree and separately restored runtime assets."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "run_saafec_stair_inference.sh",
    "scripts/cache/generate_proteinmpnn_logits_cache.py",
    "scripts/cache/generate_saprot_structure_aware_cache.py",
    "scripts/cache/sequence_variant_policy.py",
    "scripts/external/proteinmpnn_loader.py",
)

SAPROT_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)

SINGLE_CHECKPOINTS = (
    "seed_1337/best_head.pt",
    "seed_2027/best_head.pt",
    "seed_3407/best_head.pt",
)

DOUBLE_CHECKPOINT = (
    "assets/checkpoints/double_mutation/best_double_mutation_package.pt"
)
THERMOMPNN_SOURCE_SHA256 = (
    "3bbcb4342482438bb5d4ebe6509d514490dfce804617865fa55ffdcbda2fea12"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--mode", choices=("single", "double"), default="single")
    return parser.parse_args()


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, ROOT / default)).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    missing: list[str] = []

    for relative in SOURCE_FILES:
        if not (ROOT / relative).is_file():
            missing.append(str(ROOT / relative))

    if not args.source_only:
        saprot_dir = env_path(
            "SAPROT_MODEL_DIR", "assets/external/models/SaProt_650M_PDB"
        )
        for filename in SAPROT_FILES:
            if not (saprot_dir / filename).is_file():
                missing.append(str(saprot_dir / filename))

        required_files = [
            env_path(
                "PROTEINMPNN_CHECKPOINT",
                "assets/external/models/proteinmpnn/v_48_020.pt",
            ),
            env_path(
                "PROTEINMPNN_SOURCE",
                "assets/external/source/ThermoMPNN/protein_mpnn_utils.py",
            ),
            env_path("FOLDSEEK_BIN", "assets/external/bin/foldseek"),
        ]
        single_checkpoint_dir = env_path(
            "SINGLE_CHECKPOINT_DIR", "assets/checkpoints/single_mutation/seeds"
        )
        required_files.extend(single_checkpoint_dir / path for path in SINGLE_CHECKPOINTS)
        if args.mode == "double":
            required_files.append(
                env_path("DOUBLE_HEAD_CHECKPOINT", DOUBLE_CHECKPOINT)
            )
        missing.extend(str(path) for path in required_files if not path.is_file())

        foldseek = env_path("FOLDSEEK_BIN", "assets/external/bin/foldseek")
        if foldseek.is_file() and not os.access(foldseek, os.X_OK):
            missing.append(f"{foldseek} (not executable)")
        proteinmpnn_source = env_path(
            "PROTEINMPNN_SOURCE",
            "assets/external/source/ThermoMPNN/protein_mpnn_utils.py",
        )
        if (
            proteinmpnn_source.is_file()
            and sha256(proteinmpnn_source) != THERMOMPNN_SOURCE_SHA256
        ):
            missing.append(
                f"{proteinmpnn_source} (does not match the pinned ThermoMPNN source)"
            )

    if missing:
        print("Installation check failed. Missing or unusable:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    scope = "source tree" if args.source_only else f"{args.mode}-mode installation"
    print(f"OK: {scope}")


if __name__ == "__main__":
    main()
