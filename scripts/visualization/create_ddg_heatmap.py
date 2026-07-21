#!/usr/bin/env python3
"""Update a standalone all-mutant DDG heatmap HTML from a mutation matrix CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
STABILITY_THRESHOLD = 0.5
DEFAULT_TEMPLATE_NAME = "ddg_heatmap.html"


def aa_delta_delta_g_column(aa: str, fieldnames: list[str] | None) -> str | None:
    candidates = [
        f"{aa}_ΔΔG_kcal_per_mol",
        f"{aa}_DDG_kcal_per_mol",
        f"{aa}_ddg_kcal_per_mol",
    ]
    available = set(fieldnames or [])
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def read_matrix(csv_path: Path) -> tuple[list[dict], dict]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    aa_columns = {
        aa: aa_delta_delta_g_column(aa, reader.fieldnames)
        for aa in AA_ORDER
    }
    missing = [aa for aa, column in aa_columns.items() if column is None]
    if missing:
        raise ValueError(
            "Missing expected amino-acid ΔΔG columns for: "
            + ", ".join(missing)
        )

    payload_rows: list[dict] = []
    mutation_values: list[float] = []
    self_errors: list[str] = []

    for row in rows:
        wt = row["wild_type_aa"]
        values = {}
        for aa in AA_ORDER:
            value = float(row[aa_columns[aa]])
            values[aa] = value
            if aa != wt:
                mutation_values.append(value)

        self_value = values.get(wt)
        if self_value is None or abs(self_value) > 1e-9:
            self_errors.append(f"{row['sequence_index']}{wt}={self_value}")

        non_self_values = [value for aa, value in values.items() if aa != wt]
        best_aa, best_value = min(
            ((aa, value) for aa, value in values.items() if aa != wt),
            key=lambda item: item[1],
        )
        worst_aa, worst_value = max(
            ((aa, value) for aa, value in values.items() if aa != wt),
            key=lambda item: item[1],
        )

        payload_rows.append(
            {
                "index": int(row["sequence_index"]),
                "chain": row["chain"],
                "pdbResidueNumber": row["pdb_residue_number"],
                "insertionCode": row["insertion_code"],
                "wildType": wt,
                "wildTypeResidue": row["wild_type_residue"],
                "values": values,
                "mean": sum(non_self_values) / len(non_self_values),
                "bestMutation": f"{wt}{row['sequence_index']}{best_aa}",
                "bestAa": best_aa,
                "bestValue": best_value,
                "worstMutation": f"{wt}{row['sequence_index']}{worst_aa}",
                "worstAa": worst_aa,
                "worstValue": worst_value,
            }
        )

    stabilizing = sum(value <= -STABILITY_THRESHOLD for value in mutation_values)
    neutral = sum(abs(value) < STABILITY_THRESHOLD for value in mutation_values)
    destabilizing = sum(value >= STABILITY_THRESHOLD for value in mutation_values)
    strongest = min(
        ((row["bestValue"], row["bestMutation"]) for row in payload_rows),
        key=lambda item: item[0],
    )
    weakest = max(
        ((row["worstValue"], row["worstMutation"]) for row in payload_rows),
        key=lambda item: item[0],
    )

    meta = {
        "sourceCsv": csv_path.name,
        "sequence": "".join(row["wildType"] for row in payload_rows),
        "residueCount": len(payload_rows),
        "aaCount": len(AA_ORDER),
        "mutationCount": len(mutation_values),
        "minValue": min(mutation_values),
        "maxValue": max(mutation_values),
        "meanValue": sum(mutation_values) / len(mutation_values),
        "stabilizingCount": stabilizing,
        "neutralCount": neutral,
        "destabilizingCount": destabilizing,
        "strongestMutation": strongest[1],
        "strongestValue": strongest[0],
        "weakestMutation": weakest[1],
        "weakestValue": weakest[0],
        "selfMutationNonzeroCount": len(self_errors),
    }

    return payload_rows, meta


def _atom_element(line: str, atom_name: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    for character in atom_name:
        if character.isalpha():
            return character.upper()
    return "C"


def read_structure(pdb_path: Path | None, rows: list[dict]) -> dict:
    if pdb_path is None or not pdb_path.exists():
        return {
            "sourcePdb": None,
            "pdbText": "",
            "atoms": [],
            "backbone": [],
            "center": [0.0, 0.0, 0.0],
            "radius": 1.0,
            "residueCount": 0,
            "atomCount": 0,
        }

    residue_lookup = {
        (row["chain"], str(row["pdbResidueNumber"]), row.get("insertionCode", "")): row
        for row in rows
    }
    pdb_text = pdb_path.read_text()
    atoms: list[dict] = []
    backbone: list[dict] = []
    residues_seen: set[tuple[str, str, str]] = set()

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue

        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain = line[21].strip() or "_"
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        element = _atom_element(line, atom_name)
        if element == "H":
            continue

        row = residue_lookup.get((chain, residue_number, insertion_code))
        if row is None:
            continue

        atom = {
            "name": atom_name,
            "element": element,
            "residueName": residue_name,
            "chain": chain,
            "residueNumber": residue_number,
            "insertionCode": insertion_code,
            "seqIndex": row["index"],
            "wildType": row["wildType"],
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
        }
        atoms.append(atom)
        residues_seen.add((chain, residue_number, insertion_code))
        if atom_name == "CA":
            backbone.append(atom)

    if not atoms:
        center = [0.0, 0.0, 0.0]
        radius = 1.0
    else:
        center = [
            sum(atom[axis] for atom in atoms) / len(atoms)
            for axis in ("x", "y", "z")
        ]
        radius = max(
            (
                (
                    (atom["x"] - center[0]) ** 2
                    + (atom["y"] - center[1]) ** 2
                    + (atom["z"] - center[2]) ** 2
                )
                ** 0.5
                for atom in atoms
            ),
            default=1.0,
        )

    return {
        "sourcePdb": pdb_path.name,
        "pdbText": pdb_text,
        "atoms": atoms,
        "backbone": backbone,
        "center": center,
        "radius": radius,
        "residueCount": len(residues_seen),
        "atomCount": len(atoms),
    }


def normalize_display_title(title: str) -> str:
    """Use the scientific ΔΔG notation in rendered page titles."""
    return re.sub(r"\bDDG\b", "ΔΔG", title, flags=re.IGNORECASE)


def render_html(rows: list[dict], meta: dict, title: str, structure: dict, template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(
            f"{template_path} does not exist. Pass --template pointing to an existing "
            "base heatmap HTML file, or keep ddg_heatmap.html next to this script."
        )

    payload = json.dumps(
        {
            "aminoAcids": AA_ORDER,
            "rows": rows,
            "meta": meta,
            "structure": structure,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")

    escaped_title = html.escape(normalize_display_title(title))
    subtitle_html = (
        '<div class="subtitle convention" aria-label="Stability convention">\n'
        '          <span class="convention-item convention-stabilizing"><b>Stabilizing</b><em>ΔΔG ≤ -0.5 kcal/mol</em></span>\n'
        '          <span class="convention-item convention-neutral"><b>Near-Neutral</b><em>|ΔΔG| &lt; 0.5 kcal/mol</em></span>\n'
        '          <span class="convention-item convention-destabilizing"><b>Destabilizing</b><em>ΔΔG ≥ +0.5 kcal/mol</em></span>\n'
        '        </div>'
    )
    subtitle_pattern = (
        r'<(?:p|div)\s+class="subtitle(?: convention)?"[^>]*>.*?</(?:p|div)>'
    )
    html_text = template_path.read_text()

    html_text = re.sub(r"<title>.*?</title>", f"<title>{escaped_title}</title>", html_text, count=1, flags=re.S)
    html_text = re.sub(r"<h1>.*?</h1>", f"<h1>{escaped_title}</h1>", html_text, count=1, flags=re.S)
    if re.search(subtitle_pattern, html_text, flags=re.S):
        html_text = re.sub(
            subtitle_pattern,
            subtitle_html,
            html_text,
            count=1,
            flags=re.S,
        )
    else:
        html_text = re.sub(
            r"(<h1>.*?</h1>)",
            lambda match: f"{match.group(1)}\n        {subtitle_html}",
            html_text,
            count=1,
            flags=re.S,
        )
    html_text = re.sub(
        r'<script id="ddg-data" type="application/json">.*?</script>',
        lambda _match: f'<script id="ddg-data" type="application/json">{payload}</script>',
        html_text,
        count=1,
        flags=re.S,
    )
    return html_text


def default_template_path(output_html: Path, script_dir: Path) -> Path:
    master_template = script_dir / DEFAULT_TEMPLATE_NAME
    if master_template.exists():
        return master_template
    demo_template = script_dir / "2LJ3_ddg_heatmap.html"
    if demo_template.exists():
        return demo_template
    if output_html.exists():
        return output_html
    raise FileNotFoundError(
        f"No HTML template found. Expected {DEFAULT_TEMPLATE_NAME}; pass --template "
        "pointing to an existing heatmap HTML file."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input all-mutant DDG matrix CSV")
    parser.add_argument("html", type=Path, help="Output standalone HTML heatmap")
    parser.add_argument("--pdb", type=Path, default=None, help="Optional PDB file to embed")
    parser.add_argument("--template", type=Path, default=None, help="Existing HTML file to use as the heatmap base")
    parser.add_argument("--title", default=None, help="Optional page title")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    template_path = args.template or default_template_path(args.html, script_dir)
    rows, meta = read_matrix(args.csv)
    structure = read_structure(args.pdb, rows)
    title_prefix = (
        Path(structure["sourcePdb"]).stem
        if structure.get("sourcePdb")
        else args.csv.stem
    )
    title = args.title or f"{title_prefix} Mutation ΔΔG Stability Predictions"
    args.html.write_text(render_html(rows, meta, title, structure, template_path))

    print(f"Wrote {args.html}")
    print(f"Rows: {meta['residueCount']}; mutations: {meta['mutationCount']}")
    if structure.get("sourcePdb"):
        print(f"Embedded PDB: {structure['sourcePdb']} ({structure['atomCount']} heavy atoms)")


if __name__ == "__main__":
    main()
