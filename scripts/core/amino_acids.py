"""Canonical amino-acid ordering used by SAAFEC-STAIR model heads."""

AMINO_ACIDS_20: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX: dict[str, int] = {
    amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS_20)
}
NUM_AMINO_ACIDS: int = len(AMINO_ACIDS_20)
