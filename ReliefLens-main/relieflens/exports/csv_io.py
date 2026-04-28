from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


def export_csv(data: Iterable[Dict[str, object]], filename: str) -> None:
    path = Path(filename)
    rows = list(data)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def import_csv(filename: str) -> List[Dict[str, str]]:
    path = Path(filename)
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
