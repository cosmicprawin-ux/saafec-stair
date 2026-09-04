"""Shared WT-sequence canonicalisation for sequence-derived caches."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def summarise_sequence_variants(sequence_counts: Counter[str]) -> tuple[dict[str, Any], ...]:
    """Return deterministic metadata for all WT sequence variants seen for one protein."""
    return tuple(
        {
            "sequence": sequence,
            "length": len(sequence),
            "count": count,
        }
        for sequence, count in sorted(
            sequence_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
    )


def is_simple_terminal_variant(sequence_a: str, sequence_b: str) -> bool:
    """Return true when two sequences differ only by an N/C-terminal extension."""
    if sequence_a == sequence_b:
        return True
    shorter, longer = sorted((sequence_a, sequence_b), key=len)
    return longer.startswith(shorter) or longer.endswith(shorter)


def choose_canonical_sequence(
    protein_name: str,
    sequence_counts: Counter[str],
    table_path: Path,
) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    """Choose a workbook sequence while rejecting ambiguous internal conflicts."""
    if not sequence_counts:
        raise ValueError(
            f"{table_path} has no WT sequence value for {protein_name}. "
            "Expected one of wt_seq_pdb, wt_seq, WT_SEQ, wildtype_sequence, or sequence."
        )
    variant_summary = summarise_sequence_variants(sequence_counts)
    if len(sequence_counts) == 1:
        return next(iter(sequence_counts)), "single_workbook_wt_sequence", variant_summary

    canonical = sorted(
        sequence_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )[0][0]
    incompatible = [
        f"L={len(sequence)} count={count}"
        for sequence, count in sorted(
            sequence_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        if not is_simple_terminal_variant(canonical, sequence)
    ]
    if incompatible:
        raise ValueError(
            f"{protein_name} has multiple WT sequences in {table_path} that are not "
            f"simple terminal variants relative to the canonical sequence "
            f"(L={len(canonical)}): {', '.join(incompatible)}."
        )
    return (
        canonical,
        "most_frequent_workbook_wt_sequence_terminal_variant_allowed",
        variant_summary,
    )
