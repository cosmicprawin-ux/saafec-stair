from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inference/write_specified_single_mutation_predictions.py"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class SpecifiedSingleMutationOutputTests(unittest.TestCase):
    def run_export(self, root: Path, requests: str) -> Path:
        matrix_dir = root / "matrices"
        output_dir = root / "outputs"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        (matrix_dir / "1ABC_A_predicted_DDG_matrix.csv").write_text(
            "sequence_index,chain,pdb_residue_number,insertion_code,wild_type_aa,"
            "wild_type_residue,A_DDG_kcal_per_mol,V_DDG_kcal_per_mol\n"
            "1,A,1,,A,ALA,0,1.25\n",
            encoding="utf-8",
        )
        request_path = root / "requests.csv"
        request_path.write_text(requests, encoding="utf-8")
        subprocess.run(
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
            check=True,
        )
        return output_dir

    def test_clean_predictions_and_separate_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = self.run_export(
                Path(temp_dir),
                "pdb,chain,mutation\n1ABC_A,A,A1V\n1ABC_A,A,G1V\n",
            )

            prediction_path = output_dir / "specified_single_mutation_DDG_predictions.csv"
            fields, rows = read_rows(prediction_path)
            self.assertNotIn("status", fields)
            self.assertNotIn("message", fields)
            self.assertEqual([row["mutation"] for row in rows], ["A1V"])
            self.assertEqual(rows[0]["predicted_DDG_kcal_per_mol"], "1.25")

            validation_path = output_dir / "specified_single_mutation_DDG_validation_errors.csv"
            validation_fields, validation_rows = read_rows(validation_path)
            self.assertIn("status", validation_fields)
            self.assertIn("message", validation_fields)
            self.assertEqual(validation_rows[0]["status"], "wild_type_mismatch")

            summary = json.loads(
                (output_dir / "specified_single_mutation_DDG_predictions.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["n_requested"], 2)
            self.assertEqual(summary["n_predicted"], 1)
            self.assertEqual(summary["n_validation_errors"], 1)

    def test_successful_run_does_not_leave_validation_error_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = self.run_export(
                root,
                "pdb,chain,mutation\n1ABC_A,A,G1V\n",
            )
            self.assertTrue(
                (output_dir / "specified_single_mutation_DDG_validation_errors.csv").is_file()
            )

            output_dir = self.run_export(
                root,
                "pdb,chain,mutation\n1ABC_A,A,A1V\n",
            )
            self.assertFalse(
                (output_dir / "specified_single_mutation_DDG_validation_errors.csv").exists()
            )
            self.assertFalse(
                (output_dir / "specified_single_mutation_DDG_validation_errors.txt").exists()
            )


if __name__ == "__main__":
    unittest.main()
