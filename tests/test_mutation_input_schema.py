from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


if importlib.util.find_spec("torch") is None:
    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []  # type: ignore[attr-defined]
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_module.__path__ = []  # type: ignore[attr-defined]
    torch_data_module = types.ModuleType("torch.utils.data")
    torch_data_module.Dataset = object  # type: ignore[attr-defined]
    torch_module.utils = torch_utils_module  # type: ignore[attr-defined]
    torch_utils_module.data = torch_data_module  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_data_module


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core.mutation_dataset import load_mutations_csv  # noqa: E402
from core.double_mutation_dataset import load_double_mutation_delimited  # noqa: E402


class MutationInputSchemaTests(unittest.TestCase):
    def load(self, contents: str) -> dict[str, list[dict[str, object]]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mutations.csv"
            path.write_text(contents, encoding="utf-8")
            return load_mutations_csv(path)

    def test_compact_single_mutation_is_canonical_public_schema(self) -> None:
        records = self.load(
            "# Required columns: pdb (without the .pdb extension), chain, mut.\n"
            "# Columns may be separated by spaces or tabs.\n"
            "# Wild-type and mutant amino acids use standard one-letter amino-acid codes.\n"
            "# mut format: WT_AA + residue_number + mutant_AA (example: A10V).\n"
            "\n"
            "pdb\tchain\tmut\n"
            "1ABC_A\tA\tA10V\n"
        )
        mutation = records["1ABC_A"][0]
        self.assertEqual(mutation["position"], 9)
        self.assertEqual(mutation["wt_aa"], "A")
        self.assertEqual(mutation["mut_aa"], "V")

    def test_single_mutation_list_accepts_spaces_without_comments(self) -> None:
        records = self.load(
            "pdb    chain  mut\n"
            "1ABC_A A      A10V\n"
        )
        self.assertEqual(records["1ABC_A"][0]["position"], 9)

    def test_single_mutation_list_accepts_mixed_spaces_and_tabs(self) -> None:
        records = self.load(
            "pdb\tchain  mut\n"
            "1ABC_A   A\tA10V\n"
        )
        self.assertEqual(records["1ABC_A"][0]["mut_aa"], "V")

    def test_compact_double_mutation_list_has_aligned_public_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "double_mutation_list.txt"
            path.write_text(
                "# Required columns: pdb, chain, mut_1, mut_2.\n"
                "# Columns may be separated by spaces or tabs.\n"
                "\n"
                "pdb\tchain\tmut_1\tmut_2\n"
                "1ABC_A\tA\tA10V\tG25D\n",
                encoding="utf-8",
            )
            records = load_double_mutation_delimited(path)

        mutation = records["1ABC_A"].mutations[0]
        self.assertEqual(mutation.positions_raw, (10, 25))
        self.assertEqual(mutation.wt_aa, ("A", "G"))
        self.assertEqual(mutation.mt_aa, ("V", "D"))

    def test_double_mutation_list_accepts_spaces_without_comments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "double_mutation_list.txt"
            path.write_text(
                "pdb    chain mut_1   mut_2\n"
                "1ABC_A A     A10V    G25D\n",
                encoding="utf-8",
            )
            records = load_double_mutation_delimited(path)

        mutation = records["1ABC_A"].mutations[0]
        self.assertEqual(mutation.positions_raw, (10, 25))

    def test_matching_compact_and_expanded_values_are_accepted(self) -> None:
        records = self.load(
            "pdb,chain,mutation,position,wt_aa,mut_aa\n"
            "1ABC_A,A,A10V,10,A,V\n"
        )
        self.assertEqual(records["1ABC_A"][0]["position"], 9)

    def test_conflicting_mutation_representations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutation representations conflict"):
            self.load(
                "pdb,chain,mutation,position,wt_aa,mut_aa\n"
                "1ABC_A,A,A10V,11,A,V\n"
            )


if __name__ == "__main__":
    unittest.main()
