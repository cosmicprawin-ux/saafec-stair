"""Shared CSV formatting for public prediction tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, TextIO


UNIT_DECLARATION = "Unit(DDG)=kcal/mol"
LEGACY_UNIT_LABEL = "Unit"
LEGACY_UNIT_VALUE = "kcal_per_mol"


def write_prediction_rows(
    handle: TextIO,
    columns: list[tuple[str, str]],
    rows: Iterable[dict[str, Any]],
    *,
    top_labels: list[str] | None = None,
) -> None:
    """Write a unit/group row, a display-header row, and ordered prediction rows."""
    writer = csv.writer(handle, lineterminator="\n")
    width = len(columns)
    if top_labels is None:
        top_labels = [UNIT_DECLARATION, *([""] * max(0, width - 1))]
    if len(top_labels) != width:
        raise ValueError(f"Expected {width} top-row labels, received {len(top_labels)}")
    if not top_labels or top_labels[0] != UNIT_DECLARATION:
        raise ValueError(f"Prediction CSV top-left cell must be {UNIT_DECLARATION}")
    writer.writerow(top_labels)
    writer.writerow([display_name for _, display_name in columns])
    for row in rows:
        writer.writerow([row.get(source_name, "") for source_name, _ in columns])


def read_prediction_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the current two-row format, with support for legacy one-row CSVs."""
    with path.open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.reader(handle), None)
        if first_row is None:
            return [], []
        if first_row and first_row[0].strip() == UNIT_DECLARATION:
            reader = csv.DictReader(handle)
        elif first_row and first_row[0].strip() == LEGACY_UNIT_LABEL:
            if len(first_row) < 2 or first_row[1].strip() != LEGACY_UNIT_VALUE:
                raise ValueError(f"Unsupported prediction unit row in {path}")
            reader = csv.DictReader(handle)
        else:
            handle.seek(0)
            reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def format_final_ddg(value: Any) -> str:
    """Format one final user-facing DDG value without changing upstream values."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Final DDG value must be finite, received {value!r}")
    if round(numeric, 2) == 0:
        numeric = 0.0
    return f"{numeric:.2f}"
