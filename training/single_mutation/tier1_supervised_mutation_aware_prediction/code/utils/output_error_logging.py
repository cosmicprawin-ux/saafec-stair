#!/usr/bin/env python3
"""
output_error_logging.py
-----------------------
Shared tagged error logging for the cached-embedding pipeline.

Errors and warnings are both printed to the console and appended to
step-specific folders under `repository_root/output/errors/` so failures remain
visible after the run exits.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.pipeline_config import work_path_str


def infer_output_base(output_path: str | Path, default_work_dir: Path) -> Path:
    """
    Infer the shared `repository_root/output/` root from a script-specific output path.
    """
    path = Path(output_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        parts = path.parts
        if len(parts) >= 2 and parts[:2] == (default_work_dir.name, "output"):
            path = Path(*parts[2:]) if len(parts) > 2 else Path()
            resolved = (default_work_dir / "output" / path).resolve()
        elif len(parts) >= 1 and parts[0] == "output":
            path = Path(*parts[1:]) if len(parts) > 1 else Path()
            resolved = (default_work_dir / "output" / path).resolve()
        elif len(parts) >= 1 and parts[0] == default_work_dir.name:
            path = Path(*parts[1:]) if len(parts) > 1 else Path()
            resolved = (default_work_dir / path).resolve()
        else:
            resolved = (default_work_dir / path).resolve()

    current = resolved
    while True:
        if current.name == "output":
            return current
        if current.parent == current:
            break
        current = current.parent
    return (default_work_dir / "output").resolve()


@dataclass
class OutputErrorLogger:
    """Tagged error/warning logger that writes to step-specific output folders."""

    script_name: str
    output_base: Path

    def __post_init__(self) -> None:
        self.error_dir = self.output_base / "errors"
        self.error_dir.mkdir(parents=True, exist_ok=True)
        self.step_dir = self.error_dir / self.script_name
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.script_log_path = self.step_dir / f"{self.script_name}_errors.log"
        self.shared_log_path = self.error_dir / f"pipeline_errors_{self.script_name}.log"
        self.run_status_path = self.step_dir / f"{self.script_name}_{self.run_id}_status.json"

    def _write(self, line: str) -> None:
        for path in (self.script_log_path, self.shared_log_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log(
        self,
        tag: str,
        message: str,
        *,
        level: str = "ERROR",
        context: dict[str, Any] | None = None,
        echo: bool = True,
    ) -> str:
        timestamp = datetime.now().isoformat()
        parts = [f"[{level}]", f"[{tag}]", f"[{self.script_name}]"]
        if context:
            safe_context = json.dumps(context, sort_keys=True, default=str)
            line = f"{timestamp} {' '.join(parts)} {message} | context={safe_context}"
        else:
            line = f"{timestamp} {' '.join(parts)} {message}"
        self._write(line)
        if echo:
            print(f"{' '.join(parts)} {message}")
        return line

    def info(
        self,
        tag: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        echo: bool = True,
    ) -> str:
        return self.log(tag, message, level="INFO", context=context, echo=echo)

    def warning(
        self,
        tag: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        echo: bool = True,
    ) -> str:
        return self.log(tag, message, level="WARNING", context=context, echo=echo)

    def error(
        self,
        tag: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        echo: bool = True,
    ) -> str:
        return self.log(tag, message, level="ERROR", context=context, echo=echo)

    def exception(
        self,
        tag: str,
        message: str,
        exc: Exception,
        *,
        context: dict[str, Any] | None = None,
        echo: bool = True,
    ) -> str:
        extra = dict(context or {})
        extra["exception_type"] = type(exc).__name__
        extra["exception_message"] = str(exc)
        extra["traceback"] = traceback.format_exc()
        return self.log(tag, message, level="ERROR", context=extra, echo=echo)

    def write_run_status(
        self,
        status: str,
        *,
        summary: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "script_name": self.script_name,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "error_log_path": work_path_str(self.script_log_path),
        }
        if summary:
            payload.update(summary)
        with open(self.run_status_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return self.run_status_path
