import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.text import slugify

from studio.models import (
    AssetType, Book, Chapter, Notebook, ProductionAsset, ProductionUnit,
)


SERIES = {
    "M1": {
        "book_code": "B01",
        "title": "Mathematics of SRAI",
        "package": "srai_math",
        "sequence": 1,
    },
    "M2": {
        "book_code": "B02",
        "title": "Machine Learning for SRAI",
        "package": "srai_ml",
        "sequence": 2,
    },
    "M3": {
        "book_code": "B03",
        "title": "Deep Learning for SRAI",
        "package": "srai_dl",
        "sequence": 3,
    },
    "M4": {
        "book_code": "B04",
        "title": "Generative AI, LLMs and Intelligent Agents",
        "package": "srai_genai",
        "sequence": 4,
    },
    "V5": {
        "book_code": "B05",
        "title": "Enterprise MLOps and AI Production Systems",
        "package": "srai_mlops",
        "sequence": 5,
    },
    "V6": {
        "book_code": "B06",
        "title": "Decision Intelligence",
        "package": "srai_decision",
        "sequence": 6,
    },
    "V7_0": {
        "book_code": "B07",
        "title": "Applied AI Studio: National Sector Intelligence",
        "package": "srai_applied",
        "sequence": 7,
    },
    "V7_A": {
        "book_code": "B07", "title": "Applied AI Studio: National Sector Intelligence",
        "package": "srai_applied", "sequence": 7,
    },
    "V7_B": {
        "book_code": "B07", "title": "Applied AI Studio: National Sector Intelligence",
        "package": "srai_applied", "sequence": 7,
    },
    "V7_C": {
        "book_code": "B07", "title": "Applied AI Studio: National Sector Intelligence",
        "package": "srai_applied", "sequence": 7,
    },
    "V7_D": {
        "book_code": "B07", "title": "Applied AI Studio: National Sector Intelligence",
        "package": "srai_applied", "sequence": 7,
    },
    "V8": {
        "book_code": "B08",
        "title": "Enterprise AI Architecture and Platforms",
        "package": "srai_enterprise",
        "sequence": 8,
    },
    "V9": {
        "book_code": "B09",
        "title": "AI Transformation and Executive Leadership",
        "package": "srai_strategy",
        "sequence": 9,
    },
}

PATTERN = re.compile(r"^(M[1-4]|V[5689]|V7_[0ABCD])_N(\d{2})_(.+)\.ipynb$")

BOOK7_OFFSETS = {"V7_0": 0, "V7_A": 5, "V7_B": 20, "V7_C": 35, "V7_D": 50}

DEFAULT_ASSETS = [
    ("book_chapter", "Book chapter", "educational", 20),
    ("notebook", "Notebook", "educational", 20),
    ("exercises", "Exercises", "educational", 10),
    ("youtube", "Long YouTube lesson", "communication", 15),
    ("linkedin", "LinkedIn post", "communication", 10),
    ("github", "GitHub landing page", "professional", 15),
    ("executive_brief", "Executive brief", "professional", 10),
]


def notebook_title(path, fallback):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            for line in cell.get("source", []):
                if line.lstrip().startswith("# "):
                    title = line.lstrip()[2:].strip()
                    if " — " in title:
                        return title.split(" — ", 1)[1].strip()
                    return title
    except (OSError, ValueError, TypeError):
        pass
    return fallback.replace("_", " ").title()


class Command(BaseCommand):
    help = "Import the canonical nine-book, 250-notebook SRAI collection idempotently."

    def add_arguments(self, parser):
        parser.add_argument(
            "--root",
            default="content/notebooks",
            help="Directory containing canonical series subdirectories.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        root = Path(options["root"]).resolve()
        if not root.is_dir():
            raise CommandError(f"Notebook root does not exist: {root}")

        asset_types = []
        for sequence, (code, name, family, weight) in enumerate(DEFAULT_ASSETS, 1):
            asset_type, _ = AssetType.objects.update_or_create(
                code=code,
                defaults={
                    "name": name, "family": family, "required": True,
                    "weight": weight, "sequence": sequence,
                },
            )
            asset_types.append(asset_type)

        books = {}
        books_by_code = {}
        for series_code, metadata in SERIES.items():
            book_code = metadata["book_code"]
            if book_code not in books_by_code:
                book, _ = Book.objects.update_or_create(
                    code=book_code,
                    defaults={
                        "title": metadata["title"],
                        "slug": slugify(metadata["title"]),
                        "description": f"SRAI canonical notebook collection {book_code}.",
                        "edition": "1.0",
                        "status": "active",
                        "sequence": metadata["sequence"],
                    },
                )
                books_by_code[book_code] = book
            book = books_by_code[book_code]
            books[series_code] = book

        # The packaged v2.0 database contains the superseded 15-unit Book 7 and
        # no completed Book 7 production assets. Reconcile it once, while the
        # untouched source database remains preserved in the legacy archive.
        book7 = books_by_code["B07"]
        existing_book7_codes = set(
            Notebook.objects.filter(chapter__book=book7).values_list("code", flat=True)
        )
        canonical_book7_codes = {
            *{f"V7_0_N{i:02d}" for i in range(1, 6)},
            *{f"V7_A_N{i:02d}" for i in range(1, 16)},
            *{f"V7_B_N{i:02d}" for i in range(1, 16)},
            *{f"V7_C_N{i:02d}" for i in range(1, 16)},
            *{f"V7_D_N{i:02d}" for i in range(1, 16)},
        }
        if existing_book7_codes and existing_book7_codes != canonical_book7_codes:
            if existing_book7_codes == {f"V7_A_N{i:02d}" for i in range(1, 16)}:
                Chapter.objects.filter(book=book7).delete()
            else:
                raise CommandError(
                    "Book 7 contains an unrecognized partial catalogue; refusing automatic reconciliation."
                )

        imported = []
        rejected = []
        files = sorted(root.glob("*/*.ipynb"))
        for path in files:
            match = PATTERN.match(path.name)
            if not match:
                rejected.append(path.name)
                continue
            series_code, number_text, fallback = match.groups()
            source_number = int(number_text)
            number = BOOK7_OFFSETS.get(series_code, 0) + source_number
            metadata = SERIES[series_code]
            title = notebook_title(path, fallback)
            chapter_code = f"{metadata['book_code']}-C{number:02d}"
            chapter, _ = Chapter.objects.update_or_create(
                code=chapter_code,
                defaults={
                    "book": books[series_code],
                    "number": number,
                    "title": title,
                    "slug": slugify(f"{number:02d}-{title}"),
                    "status": "active",
                    "estimated_study_minutes": 90,
                },
            )
            notebook_code = f"{series_code}_N{source_number:02d}"
            relative_path = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
            Notebook.objects.update_or_create(
                chapter=chapter,
                defaults={
                    "code": notebook_code,
                    "title": title,
                    "repository_path": str(relative_path),
                    "package_name": metadata["package"],
                    "validation_status": "not_tested",
                },
            )
            unit, _ = ProductionUnit.objects.get_or_create(
                chapter=chapter,
                defaults={
                    "code": f"PU-{chapter_code}",
                    "workflow_state": "planned",
                    "priority": "normal",
                    "next_action": "Validate notebook and complete Gold Standard assets",
                },
            )
            for asset_type in asset_types:
                ProductionAsset.objects.get_or_create(
                    production_unit=unit,
                    asset_type=asset_type,
                    defaults={
                        "title": f"{title} — {asset_type.name}",
                        "status": "not_started",
                    },
                )
            imported.append(notebook_code)

        expected = {
            *{f"M1_N{i:02d}" for i in range(1, 31)},
            *{f"M2_N{i:02d}" for i in range(1, 36)},
            *{f"M3_N{i:02d}" for i in range(1, 31)},
            *{f"M4_N{i:02d}" for i in range(1, 31)},
            *{f"V5_N{i:02d}" for i in range(1, 16)},
            *{f"V6_N{i:02d}" for i in range(1, 16)},
            *{f"V7_0_N{i:02d}" for i in range(1, 6)},
            *{f"V7_A_N{i:02d}" for i in range(1, 16)},
            *{f"V7_B_N{i:02d}" for i in range(1, 16)},
            *{f"V7_C_N{i:02d}" for i in range(1, 16)},
            *{f"V7_D_N{i:02d}" for i in range(1, 16)},
            *{f"V8_N{i:02d}" for i in range(1, 16)},
            *{f"V9_N{i:02d}" for i in range(1, 16)},
        }
        actual = set(imported)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected or rejected or len(actual) != 250:
            raise CommandError(
                f"Catalog validation failed: imported={len(actual)}, missing={missing}, "
                f"unexpected={unexpected}, rejected={rejected}"
            )
        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {len(actual)} canonical notebooks into {len(books_by_code)} books; "
                f"{ProductionUnit.objects.count()} Production Units available."
            )
        )
