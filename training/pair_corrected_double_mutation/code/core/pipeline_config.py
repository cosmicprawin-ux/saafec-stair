"""Path helpers shared by the selected-model training workflows.

External data, structures, third-party source, executables, and weights are
intentionally kept outside this release. Command-line paths take precedence;
the command-line entrypoints use environment variables and repository-relative
asset locations as documented in the top-level training README.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = SCRIPTS_DIR.parent
TRAINING_DIR = next(
    (parent for parent in WORK_DIR.parents if parent.name == "training"),
    WORK_DIR.parent,
)
REPO_ROOT = TRAINING_DIR.parent
ASSETS_DIR = REPO_ROOT / "assets"
EXTERNAL_ASSETS_DIR = ASSETS_DIR / "external"

DEFAULT_DATABASE_DIR = Path(
    os.environ.get("SAAFEC_TRAINING_DATA_ROOT", REPO_ROOT / "data")
).expanduser()
DEFAULT_OUTPUT_ROOT = WORK_DIR / "output"
MODEL_DIR = EXTERNAL_ASSETS_DIR / "models"
DEFAULT_SAPROT_MODEL_DIR = Path(
    os.environ.get("SAPROT_MODEL_DIR", MODEL_DIR / "SaProt_650M_PDB")
).expanduser()
DEFAULT_FOLDSEEK_BIN = Path(
    os.environ.get("FOLDSEEK_BIN", EXTERNAL_ASSETS_DIR / "bin" / "foldseek")
).expanduser()
DEFAULT_PROTEINMPNN_CHECKPOINT = Path(
    os.environ.get(
        "PROTEINMPNN_CHECKPOINT", MODEL_DIR / "proteinmpnn" / "v_48_020.pt"
    )
).expanduser()
DEFAULT_PROTEINMPNN_SOURCE = Path(
    os.environ.get(
        "PROTEINMPNN_SOURCE",
        EXTERNAL_ASSETS_DIR / "source" / "ThermoMPNN" / "protein_mpnn_utils.py",
    )
).expanduser()
DEFAULT_STRUCTURE_ROOT = Path(
    os.environ.get("STRUCTURE_ROOT", REPO_ROOT / "structures")
).expanduser()
DEFAULT_SINGLE_CHECKPOINT_DIR = Path(
    os.environ.get(
        "SINGLE_CHECKPOINT_DIR", ASSETS_DIR / "checkpoints" / "single_mutation" / "seeds"
    )
).expanduser()

def resolve_path_with_base(
    raw_path: str | Path,
    *,
    base_dir: Path,
    strip_prefixes: tuple[tuple[str, ...], ...] = (),
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    for prefix in strip_prefixes:
        if tuple(parts[: len(prefix)]) == prefix:
            path = Path(*parts[len(prefix) :])
            break
    return (base_dir / path).resolve()


def resolve_work_path(raw_path: str | Path) -> Path:
    return resolve_path_with_base(
        raw_path,
        base_dir=WORK_DIR,
        strip_prefixes=((WORK_DIR.name,),),
    )


def resolve_output_path(raw_path: str | Path) -> Path:
    return resolve_path_with_base(
        raw_path,
        base_dir=DEFAULT_OUTPUT_ROOT,
        strip_prefixes=((WORK_DIR.name, "output"), ("output",), (WORK_DIR.name,)),
    )


def resolve_model_path(raw_path: str | Path) -> Path:
    return resolve_path_with_base(
        raw_path,
        base_dir=MODEL_DIR,
        strip_prefixes=(("assets", "external", "models"), ("models",)),
    )


def work_path_str(raw_path: str | Path) -> str:
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else resolve_work_path(path)
    for base in (REPO_ROOT, WORK_DIR):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def ensure_output_root(output_root: str | Path) -> Path:
    resolved = resolve_output_path(output_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device_arg


def list_pdb_files(pdb_dir: Path) -> list[Path]:
    pdb_files = set(pdb_dir.glob("*.pdb")) | set(pdb_dir.glob("*.PDB"))
    return sorted(path.resolve() for path in pdb_files if path.is_file())
