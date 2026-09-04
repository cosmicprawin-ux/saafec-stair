from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inference/write_specified_single_mutation_predictions.py"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        unit_row = next(csv.reader(handle))
        if unit_row[0] != "Unit(DDG)=kcal/mol" or any(unit_row[1:]):
            raise AssertionError(f"Unexpected unit row: {unit_row}")
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class SpecifiedSingleMutationOutputTests(unittest.TestCase):
    def run_export(
        self,
        root: Path,
        requests: str,
        *,
        check: bool = True,
    ) -> Path | subprocess.CompletedProcess[str]:
        matrix_dir = root / "matrices"
        output_dir = root / "outputs"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        (matrix_dir / "1ABC_A_predicted_DDG_matrix.csv").write_text(
            "Unit(DDG)=kcal/mol,,,,,mut_aa_DDG,mut_aa_DDG\n"
            "sequence_index,chain,pdb_residue_number,wild_type_aa,"
            "wild_type_residue,A,V\n"
            "1,A,1,A,ALA,0.00,1.2567\n",
            encoding="utf-8",
        )
        request_path = root / "requests.csv"
        request_path.write_text(requests, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--specified-table",
                str(request_path),
                "--matrix-dir",
                str(matrix_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            check=check,
            capture_output=not check,
            text=True,
        )
        return output_dir if check else completed

    def test_prediction_csv_has_clean_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = self.run_export(
                Path(temp_dir),
                "pdb chain mut\n1ABC_A A A1V\n",
            )

            prediction_path = output_dir / "specified_single_mutation_DDG_predictions.csv"
            fields, rows = read_rows(prediction_path)
            self.assertNotIn("status", fields)
            self.assertNotIn("message", fields)
            self.assertNotIn("insertion_code", fields)
            self.assertEqual(fields[-2:], ["", "Predicted_DDG"])
            self.assertEqual([row["mutation"] for row in rows], ["A1V"])
            self.assertEqual(rows[0]["Predicted_DDG"], "1.26")
            self.assertFalse(prediction_path.with_suffix(".txt").exists())

    def test_invalid_request_fails_instead_of_writing_status_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed = self.run_export(
                root,
                "# mut format: WT_AA + residue_number + mutant_AA\n"
                "pdb\tchain\tmut\n1ABC_A\tA\tG1V\n",
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requested wild type G", completed.stderr)
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
