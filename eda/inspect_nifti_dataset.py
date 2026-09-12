#!/usr/bin/env python3
"""Inspect dimensions and metadata for a directory of NIfTI files.

Only NIfTI headers are read, so this does not load entire CT volumes into
memory. Install the dependency with: python3 -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import nibabel as nib
    from nibabel.fileholders import FileHolder
except ImportError:
    print(
        "Missing dependency: nibabel. Install it with "
        "'python3 -m pip install -r requirements.txt'.",
        file=sys.stderr,
    )
    raise SystemExit(2)


NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass
class NiftiRecord:
    """Header-level information for one NIfTI image."""

    file: str
    subject: str
    kind: str
    shape: tuple[int, ...]
    voxel_spacing_mm: tuple[float, ...]
    physical_size_mm: tuple[float, ...]
    dimensions: int
    dtype: str


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def image_kind(path: Path) -> str:
    """Classify common image names while retaining useful names for others."""
    name = path.name.lower()
    if name.startswith("mask"):
        return "mask"
    if name.startswith("orig"):
        return "orig"
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def load_header(path: Path) -> tuple[tuple[int, ...], Any]:
    """Read shape and header, including gzip-compressed files named `.nii`."""
    with path.open("rb") as stream:
        is_gzip = stream.read(2) == b"\x1f\x8b"

    if is_gzip:
        file_map = nib.Nifti1Image.make_file_map()
        with gzip.open(path, "rb") as stream:
            file_map["image"] = FileHolder(fileobj=stream, filename=str(path))
            image = nib.Nifti1Image.from_file_map(file_map)
            return tuple(int(value) for value in image.shape), image.header.copy()

    image = nib.load(str(path))
    return tuple(int(value) for value in image.shape), image.header


def inspect_file(path: Path, root: Path) -> NiftiRecord:
    shape, header = load_header(path)
    spacing = tuple(round(float(value), 3) for value in header.get_zooms()[: len(shape)])
    physical_size = tuple(
        round(size * voxel, 3) for size, voxel in zip(shape, spacing)
    )
    subject = next(
        (part for part in path.relative_to(root).parts if part.lower().startswith("subject")),
        path.parent.name,
    )
    return NiftiRecord(
        file=str(path.relative_to(root)),
        subject=subject,
        kind=image_kind(path),
        shape=shape,
        voxel_spacing_mm=spacing,
        physical_size_mm=physical_size,
        dimensions=len(shape),
        dtype=str(header.get_data_dtype()),
    )


def collect_records(root: Path) -> tuple[list[NiftiRecord], list[str]]:
    records: list[NiftiRecord] = []
    errors: list[str] = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and is_nifti(path)):
        try:
            records.append(inspect_file(path, root))
        except Exception as exc:  # Keep one malformed file from hiding the rest.
            errors.append(f"{path}: {exc}")
    return records, errors


def shape_text(values: Iterable[Any]) -> str:
    return "x".join(str(value) for value in values)


def print_table(records: list[NiftiRecord]) -> None:
    columns = ("subject", "kind", "shape", "spacing (mm)", "size (mm)", "dtype", "file")
    rows = [
        (
            record.subject,
            record.kind,
            shape_text(record.shape),
            shape_text(record.voxel_spacing_mm),
            shape_text(record.physical_size_mm),
            record.dtype,
            record.file,
        )
        for record in records
    ]
    widths = [max(len(str(value)) for value in [column] + [row[index] for row in rows]) for index, column in enumerate(columns)]
    print("  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_summary(records: list[NiftiRecord]) -> None:
    print(f"\nFiles: {len(records)}")
    print(f"Subjects: {len({record.subject for record in records})}")
    for kind in sorted({record.kind for record in records}):
        kind_records = [record for record in records if record.kind == kind]
        shapes = sorted({shape_text(record.shape) for record in kind_records})
        print(f"{kind}: {len(kind_records)} files; shapes: {', '.join(shapes)}")


def write_records(records: list[NiftiRecord], output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n")
        return

    fieldnames = list(asdict(records[0]).keys()) if records else [field.name for field in NiftiRecord.__dataclass_fields__.values()]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["shape"] = shape_text(record.shape)
            row["voxel_spacing_mm"] = shape_text(record.voxel_spacing_mm)
            row["physical_size_mm"] = shape_text(record.physical_size_mm)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing NIfTI files")
    parser.add_argument("--output", type=Path, help="Write records to a .csv or .json file")
    parser.add_argument("--format", choices=("csv", "json"), help="Output format; inferred from --output when omitted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        print(f"Dataset directory does not exist: {root}", file=sys.stderr)
        return 1

    records, errors = collect_records(root)
    if records:
        print_table(records)
        print_summary(records)
    else:
        print(f"No NIfTI files found under {root}")

    if args.output:
        fmt = args.format or ("json" if args.output.suffix.lower() == ".json" else "csv")
        write_records(records, args.output, fmt)
        print(f"\nWrote {fmt.upper()} output to {args.output}")

    if errors:
        print(f"\nSkipped {len(errors)} file(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
    return 0 if records or not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
