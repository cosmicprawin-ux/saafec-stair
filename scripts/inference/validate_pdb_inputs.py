#!/usr/bin/env python3
"""Validate inference PDB files before feature generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from inference.prepare_single_pdb_input import (
    PDBInsertionCodeError,
    validate_no_insertion_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", required=True)
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise SystemExit(f"Input structure validation failed: no .pdb files found in {pdb_dir}")

    failures: list[str] = []
    for pdb_path in pdb_files:
        try:
            validate_no_insertion_codes(pdb_path)
        except PDBInsertionCodeError as exc:
            failures.append(str(exc))

    if failures:
        raise SystemExit("\n\n".join(failures))
    print(f"Input structure validation passed: {len(pdb_files)} PDB file(s).")


if __name__ == "__main__":
    main()
