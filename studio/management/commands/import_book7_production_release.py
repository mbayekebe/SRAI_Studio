import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from studio.models import AssetType, Book, ProductionAsset, ProductionUnit, QualityReview, Release


ASSET_TYPES = [
    ("book_chapter", "Book chapter", "educational", True, 20, 1),
    ("notebook", "Notebook", "educational", True, 20, 2),
    ("exercises", "Exercises", "educational", True, 10, 3),
    ("youtube", "Long YouTube lesson", "communication", True, 15, 4),
    ("linkedin", "LinkedIn post", "communication", True, 10, 5),
    ("github", "GitHub landing page", "professional", True, 15, 6),
    ("executive_brief", "Executive brief", "professional", True, 10, 7),
    ("presentation", "Lesson presentation", "educational", False, 0, 8),
    ("video_run_of_show", "Presenter run of show", "communication", False, 0, 9),
]


class Command(BaseCommand):
    help = "Import the validated Book 7 v1.0 Production Unit release idempotently."

    def add_arguments(self, parser):
        parser.add_argument(
            "--manifest",
            default="content/book_07_import/catalogue/BOOK7_STUDIO_IMPORT_MANIFEST.json",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        manifest_path = Path(options["manifest"]).resolve()
        if not manifest_path.is_file():
            raise CommandError(f"Import manifest does not exist: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        units = payload.get("production_units", [])
        if len(units) != 65 or len({row["production_unit"] for row in units}) != 65:
            raise CommandError("Manifest must contain exactly 65 unique Production Units.")

        try:
            book = Book.objects.get(code="B07")
        except Book.DoesNotExist as exc:
            raise CommandError("Book B07 is absent. Run import_srai_catalog first.") from exc

        types = {}
        for code, name, family, required, weight, sequence in ASSET_TYPES:
            types[code], _ = AssetType.objects.update_or_create(
                code=code,
                defaults={"name": name, "family": family, "required": required,
                          "weight": weight, "sequence": sequence},
            )

        imported = []
        for row in units:
            try:
                unit = ProductionUnit.objects.select_related("chapter__book").get(
                    code=row["production_unit"], chapter__book=book
                )
            except ProductionUnit.DoesNotExist as exc:
                raise CommandError(
                    f"Missing {row['production_unit']}; run import_srai_catalog first."
                ) from exc
            if unit.chapter.notebook.code != row["canonical_notebook_id"]:
                raise CommandError(f"Notebook mismatch for {unit.code}.")

            for asset in row["assets"]:
                ProductionAsset.objects.update_or_create(
                    production_unit=unit,
                    asset_type=types[asset["asset_type"]],
                    defaults={
                        "title": asset["title"], "status": "approved", "version": "1.0",
                        "file_path": asset["primary_path"],
                        "notes": "Validated companion files: " + ", ".join(asset.get("companion_paths", [])),
                    },
                )
            unit.workflow_state = "ready"
            unit.priority = "normal"
            unit.next_action = "Point 5: final user-facing SRAI Studio test"
            unit.notes = "Book 7 controlled release v1.0; Point 4 import-ready assets installed."
            unit.save()
            unit.recalculate_completion()
            for review_type in ("technical", "educational", "editorial", "publication"):
                QualityReview.objects.update_or_create(
                    production_unit=unit, asset=None, review_type=review_type,
                    defaults={"score": 100, "passed": True,
                              "findings": "Passed in consolidated Book 7 release validation.",
                              "reviewed_at": timezone.now()},
                )
            imported.append(unit)

        release, _ = Release.objects.update_or_create(
            code="B07-V1.0",
            defaults={"name": "Book 7 — Applied AI Studio: National Sector Intelligence",
                      "version": "1.0", "status": "candidate",
                      "notes": "Point 4 import complete; Point 5 user-facing test remains."},
        )
        release.production_units.set(imported)
        if len(imported) != 65 or any(not unit.gold_standard for unit in imported):
            raise CommandError("Post-import readiness gate failed.")
        self.stdout.write(self.style.SUCCESS(
            "Imported 65 validated Book 7 Production Units; release B07-V1.0 is a candidate for Point 5."
        ))
