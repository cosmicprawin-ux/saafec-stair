"""Load the pinned, user-supplied ProteinMPNN implementation."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType


THERMOMPNN_COMMIT = "13569795daa7689b6a6df0279b383e08c6212e79"
PROTEINMPNN_SOURCE_SHA256 = (
    "3bbcb4342482438bb5d4ebe6509d514490dfce804617865fa55ffdcbda2fea12"
)


def _verify_source(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PROTEINMPNN_SOURCE_SHA256:
        raise RuntimeError(
            "ProteinMPNN source checksum mismatch for "
            f"{path}. Expected the protein_mpnn_utils.py file from ThermoMPNN "
            f"commit {THERMOMPNN_COMMIT} with SHA-256 "
            f"{PROTEINMPNN_SOURCE_SHA256}, but found {digest}."
        )


def _load_module(source_path: str | Path) -> ModuleType:
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "ProteinMPNN source module not found at "
            f"{path}. Restore the pinned ThermoMPNN protein_mpnn_utils.py "
            "described in the top-level training README."
        )
    _verify_source(path)
    spec = importlib.util.spec_from_file_location(
        "saafec_external_protein_mpnn_utils", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_proteinmpnn_class(source_path: str | Path) -> type:
    """Return ProteinMPNN from the verified upstream utility module."""
    module = _load_module(source_path)
    proteinmpnn_class = getattr(module, "ProteinMPNN", None)
    if not isinstance(proteinmpnn_class, type):
        raise ImportError(f"ProteinMPNN class was not found in {Path(source_path)}")
    return proteinmpnn_class
