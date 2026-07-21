"""Load the externally supplied ProteinMPNN implementation used by inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module(source_path: str | Path) -> ModuleType:
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "ProteinMPNN source module not found at "
            f"{path}. Restore the pinned ThermoMPNN protein_mpnn_utils.py "
            "described in docs/EXTERNAL_ASSETS.md."
        )
    spec = importlib.util.spec_from_file_location("saafec_external_protein_mpnn_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_proteinmpnn_class(source_path: str | Path) -> type:
    """Return ProteinMPNN from a user-restored upstream utility module."""
    module = _load_module(source_path)
    proteinmpnn_class = getattr(module, "ProteinMPNN", None)
    if not isinstance(proteinmpnn_class, type):
        raise ImportError(f"ProteinMPNN class was not found in {Path(source_path)}")
    return proteinmpnn_class
