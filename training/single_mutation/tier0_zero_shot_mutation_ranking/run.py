#!/usr/bin/env python3
"""Command-line entrypoint for Tier 0 zero-shot mutation ranking."""
from pathlib import Path
import runpy

runpy.run_path(
    str(
        Path(__file__).resolve().parent
        / "code/pipelines/workflow.py"
    ),
    run_name="__main__",
)
