#!/usr/bin/env python3
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from studio.models import Notebook


def main():
    destination = PROJECT_ROOT / "release" / "catalog.csv"
    rows = Notebook.objects.select_related("chapter__book").order_by(
        "chapter__book__sequence", "chapter__number"
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "notebook_code",
                "chapter_order",
                "chapter_title",
                "file_path",
                "validation_status",
            ]
        )
        for notebook in rows:
            writer.writerow(
                [
                    notebook.chapter.book.code,
                    notebook.code,
                    notebook.chapter.number,
                    notebook.chapter.title,
                    notebook.repository_path,
                    notebook.validation_status,
                ]
            )
    print(f"Exported {rows.count()} records to {destination}")


if __name__ == "__main__":
    main()
