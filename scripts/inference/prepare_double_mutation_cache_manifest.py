#!/usr/bin/env python3
"""Prepare a cache manifest for specified double-mutation inference inputs."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.double_mutation_dataset import (  # noqa: E402
    DOUBLE_SHEET,
    load_double_mutation_workbook,
    resolve_work_path,
)
from core.pipeline_config import work_path_str  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the per-protein manifest used by SaProt/ProteinMPNN cache "
            "generation for a specified double-mutation input table."
        )
    )
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--table-sheet", dest="table_sheet", default=DOUBLE_SHEET)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--pdb-link-dir", required=True)
    return parser.parse_args()


def pdb_lookup(pdb_dir: Path) -> dict[str, Path]:
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    output: dict[str, Path] = {}
    for path in pdb_dir.glob("*.pdb"):
        output[path.stem] = path
        output[path.stem.lower()] = path
    return output


def main() -> None:
    args = parse_args()
    input_path = resolve_work_path(args.input_table)
    pdb_dir = resolve_work_path(args.pdb_dir)
    manifest_path = resolve_work_path(args.manifest_path)
    pdb_link_dir = resolve_work_path(args.pdb_link_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_link_dir.mkdir(parents=True, exist_ok=True)

    records = load_double_mutation_workbook(input_path, sheet_name=args.table_sheet)
    lookup = pdb_lookup(pdb_dir)
    missing_pdbs: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for protein_name, record in sorted(records.items()):
        source_pdb = (
            lookup.get(record.pdb)
            or lookup.get(record.pdb.lower())
            or lookup.get(protein_name)
            or lookup.get(protein_name.lower())
        )
        if source_pdb is None:
            missing_pdbs.append(
                {
                    "protein_name": protein_name,
                    "pdb": record.pdb,
                    "chain": record.chain,
                }
            )
            continue
        cache_pdb_path = pdb_link_dir / f"{protein_name}.pdb"
        if cache_pdb_path.exists() or cache_pdb_path.is_symlink():
            cache_pdb_path.unlink()
        shutil.copy2(source_pdb.resolve(), cache_pdb_path)
        rows.append(
            {
                "protein_name": protein_name,
                "pdb": record.pdb,
                "chain": record.chain,
                "source_pdb": work_path_str(source_pdb),
                "pdb_link": work_path_str(cache_pdb_path),
                "n_specified_double_mutations": len(record.mutations),
            }
        )

    if missing_pdbs:
        preview = "; ".join(f"{row['protein_name']}->{row['pdb']}" for row in missing_pdbs[:10])
        suffix = "" if len(missing_pdbs) <= 10 else f"; ... and {len(missing_pdbs) - 10} more"
        raise FileNotFoundError(f"Missing PDB files for specified proteins: {preview}{suffix}")

    fieldnames = [
        "protein_name",
        "pdb",
        "chain",
        "source_pdb",
        "pdb_link",
        "n_specified_double_mutations",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "input_table": work_path_str(input_path),
        "table_sheet": args.table_sheet,
        "pdb_dir": work_path_str(pdb_dir),
        "manifest_path": work_path_str(manifest_path),
        "pdb_link_dir": work_path_str(pdb_link_dir),
        "proteins": len(rows),
        "specified_double_mutations": sum(int(row["n_specified_double_mutations"]) for row in rows),
        "note": "This manifest covers only proteins and mutations listed in the input table.",
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
