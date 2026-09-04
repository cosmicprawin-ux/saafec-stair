"""Structure identifiers derived from public PDB and chain inputs."""
from __future__ import annotations

from pathlib import Path


def pdb_stem(value: object) -> str:
    """Return a PDB filename stem from a path, filename, or stem value."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("PDB value cannot be empty.")
    name = Path(text).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def structure_key(pdb: object, chain: object) -> str:
    """Return the internal cache key for a PDB/chain pair.

    A stem that already ends in the chain suffix is retained; otherwise the
    chain is appended. For example, both (1EY0_A, A) and (1EY0, A) resolve to
    1EY0_A.
    """
    stem = pdb_stem(pdb)
    chain_text = str(chain or "").strip()
    if not chain_text:
        raise ValueError(f"Chain value cannot be empty for PDB {stem!r}.")
    suffix = f"_{chain_text}"
    return stem if stem.upper().endswith(suffix.upper()) else f"{stem}{suffix}"
