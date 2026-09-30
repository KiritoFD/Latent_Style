from __future__ import annotations

import csv
from pathlib import Path


def normalize_csv_fieldname(name: object) -> str:
    return str(name).lstrip("\ufeff").strip().strip('"')


def normalize_csv_row(row: dict[object, object]) -> dict[str, str]:
    return {normalize_csv_fieldname(key): value for key, value in row.items()}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [normalize_csv_row(row) for row in rows]


def manifest_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def write_csv_rows(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = manifest_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
