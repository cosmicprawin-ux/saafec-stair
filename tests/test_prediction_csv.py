from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inference.prediction_csv import format_final_ddg, write_prediction_rows  # noqa: E402
from visualization.create_ddg_heatmap import AA_ORDER, build_matrix_payload  # noqa: E402


class PredictionCsvTests(unittest.TestCase):
    def test_final_ddg_is_formatted_to_exactly_two_decimal_places(self) -> None:
        self.assertEqual(format_final_ddg(1.2567), "1.26")
        self.assertEqual(format_final_ddg(-1.254), "-1.25")
        self.assertEqual(format_final_ddg(2), "2.00")
        self.assertEqual(format_final_ddg(-0.001), "0.00")

    def test_matrix_headers_separate_mutant_label_from_amino_acid(self) -> None:
        handle = io.StringIO()
        columns = [
            ("sequence_index", "sequence_index"),
            ("chain", "chain"),
            ("wild_type_aa", "wild_type_aa"),
            ("A_DDG", "A"),
        ]
        write_prediction_rows(
            handle,
            columns,
            [{"sequence_index": 1, "chain": "A", "wild_type_aa": "G", "A_DDG": "0.12"}],
            top_labels=["Unit(DDG)=kcal/mol", "", "", "mut_aa_DDG"],
        )
        rows = list(csv.reader(io.StringIO(handle.getvalue())))
        self.assertEqual(
            rows[0],
            ["Unit(DDG)=kcal/mol", "", "", "mut_aa_DDG"],
        )
        self.assertEqual(
            rows[1],
            ["sequence_index", "chain", "wild_type_aa", "A"],
        )

    def test_heatmap_calculations_keep_full_precision(self) -> None:
        row = {
            "sequence_index": "1",
            "chain": "A",
            "pdb_residue_number": "1",
            "wild_type_aa": "A",
            "wild_type_residue": "ALA",
            **{f"{aa}_DDG": (0.0 if aa == "A" else 1.2567) for aa in AA_ORDER},
        }
        _, metadata = build_matrix_payload(
            [row],
            list(row),
            "full_precision_matrix.csv",
        )
        self.assertEqual(metadata["maxValue"], 1.2567)


if __name__ == "__main__":
    unittest.main()
