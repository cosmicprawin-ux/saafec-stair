"""Readers for public text mutation lists and compatible CSV inputs."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import TextIO


def mutation_list_reader(handle: TextIO, path: str | Path) -> csv.DictReader:
    """Ignore comments and accept commas or runs of spaces and tabs."""
    content_lines = (
        line
        for line in handle
        if line.strip() and not line.lstrip().startswith("#")
    )

    with Path(path).open(encoding="utf-8") as source:
        header = next(
            (
                line
                for line in source
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )

    if "," in header:
        return csv.DictReader(content_lines, delimiter=",", skipinitialspace=True)

    normalized_lines = (
        re.sub(r"[ \t]+", "\t", line.strip()) + "\n"
        for line in content_lines
    )
    return csv.DictReader(normalized_lines, delimiter="\t")


def cleaned_rows(reader: csv.DictReader) -> list[dict[str, str]]:
    """Materialize a table while trimming keys and scalar values."""
    return [
        {
            str(key or "").strip(): str(value or "").strip()
            for key, value in row.items()
        }
        for row in reader
    ]
