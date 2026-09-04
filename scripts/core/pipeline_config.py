#!/usr/bin/env python3
"""Shared paths and filesystem helpers for SAAFEC-STAIR inference."""
from __future__ import annotations

from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = SCRIPTS_DIR.parent
DEFAULT_PDB_DIR = WORK_DIR / "pdb"
DEFAULT_OUTPUT_ROOT = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "assets" / "external" / "models"


def _strip_prefix(path: Path, prefixes: tuple[tuple[str, ...], ...]) -> Path:
    parts = path.parts
    for prefix in prefixes:
        if len(parts) >= len(prefix) and tuple(parts[: len(prefix)]) == prefix:
            remainder = parts[len(prefix) :]
            return Path(*remainder) if remainder else Path()
    return path


def resolve_path_with_base(
    raw_path: str | Path,
    *,
    base_dir: Path,
    strip_prefixes: tuple[tuple[str, ...], ...] = (),
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / _strip_prefix(path, strip_prefixes)).resolve()


def work_path_str(raw_path: str | Path | None) -> str:
    if raw_path is None:
        return ""
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else resolve_work_path(path)
    for root in (WORK_DIR, WORK_DIR.parent):
        try:
            rel = resolved.relative_to(root)
            if root == WORK_DIR:
                return str(Path(WORK_DIR.name) / rel)
            return str(rel)
        except ValueError:
            continue
    if not path.is_absolute():
        return str(path)
    return path.name


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
        strip_prefixes=(
            (WORK_DIR.name, "output"),
            ("output",),
            (WORK_DIR.name,),
        ),
    )


def resolve_model_path(raw_path: str | Path) -> Path:
    return resolve_path_with_base(
        raw_path,
        base_dir=MODEL_DIR,
        strip_prefixes=(
            (WORK_DIR.name, "assets", "external", "models"),
            ("assets", "external", "models"),
            (WORK_DIR.name,),
        ),
    )


def ensure_output_root(output_root: str | Path) -> Path:
    resolved = resolve_output_path(output_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def list_pdb_files(pdb_dir: Path) -> list[Path]:
    pdb_files = set(pdb_dir.glob("*.pdb")) | set(pdb_dir.glob("*.PDB"))
    return sorted(path.resolve() for path in pdb_files if path.is_file())
