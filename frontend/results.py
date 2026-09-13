"""Discovery and loading helpers for local benchmark CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path


def discover_csv_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def read_csv_file(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], []
        rows = list(reader)
        return list(reader.fieldnames), rows
