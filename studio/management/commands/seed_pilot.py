from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.utils.text import slugify

from studio.models import (
    AssetType, Book, Chapter, Notebook, ProductionAsset, ProductionTask,
    ProductionUnit, QualityReview, Release,
)


class Command(BaseCommand):
    help = "Load the idempotent SRAI Studio Book 1, Chapters 1–5 pilot."

    def handle(self, *args, **options):
        book, _ = Book.objects.update_or_create(
            code="B01",
            defaults={
                "title": "Mathematical Foundations for SRAI",
                "slug": "mathematical-foundations-for-srai",
                "sequence": 1,
            },
        )
        chapter_titles = [
            "Why Mathematics Is the Language of AI",
            "Vectors",
            "Matrices",
            "Systems and Transformations",
            "Vector Spaces, Basis, Rank and Projections",
        ]
        asset_specs = [
            ("book_chapter", "Book chapter", "educational", 20),
            ("notebook", "Notebook", "educational", 20),
            ("exercises", "Exercises", "educational", 10),
            ("youtube", "Long YouTube lesson", "communication", 15),
            ("linkedin", "LinkedIn post", "communication", 10),
            ("github", "GitHub landing page", "professional", 15),
            ("executive_brief", "Executive brief", "professional", 10),
        ]
        asset_types = []
        for sequence, (code, name, family, weight) in enumerate(asset_specs, 1):
            asset_type, _ = AssetType.objects.update_or_create(
                code=code,
                defaults={"name": name, "family": family, "required": True, "weight": weight, "sequence": sequence},
            )
            asset_types.append(asset_type)

        units = []
        for number, title in enumerate(chapter_titles, 1):
            code = f"B01-C{number:02d}"
            chapter, _ = Chapter.objects.update_or_create(
                code=code,
                defaults={
                    "book": book, "number": number, "title": title,
                    "slug": slugify(title), "status": "active" if number <= 2 else "planned",
                },
            )
            Notebook.objects.update_or_create(
                chapter=chapter,
                defaults={
                    "code": f"M1_N{number:02d}", "title": title,
                    "repository_path": f"notebooks/M1/M1_N{number:02d}.ipynb",
                    "package_name": "srai_math",
                },
            )
            unit, _ = ProductionUnit.objects.update_or_create(
                chapter=chapter,
                defaults={
                    "code": f"PU-{code}",
                    "workflow_state": "published" if number == 1 else "drafting" if number == 2 else "planned",
                    "priority": "high" if number <= 2 else "normal",
                    "due_date": timezone.localdate() + timedelta(days=number * 7),
                    "next_action": "Maintain Gold Standard" if number == 1 else "Complete required assets",
                },
            )
            units.append(unit)
            for asset_type in asset_types:
                status = "published" if number == 1 else "drafting" if number == 2 and asset_type.family == "educational" else "not_started"
                ProductionAsset.objects.update_or_create(
                    production_unit=unit,
                    asset_type=asset_type,
                    defaults={"title": f"{title} — {asset_type.name}", "status": status},
                )
            ProductionTask.objects.get_or_create(
                production_unit=unit,
                title=f"Advance {code} to next workflow state",
                defaults={"priority": unit.priority, "due_date": unit.due_date},
            )

        gold = units[0]
        for review_type in ["technical", "educational", "editorial", "publication"]:
            QualityReview.objects.update_or_create(
                production_unit=gold,
                review_type=review_type,
                defaults={"score": 95, "passed": True, "findings": "Pilot Gold Standard accepted."},
            )
        release, _ = Release.objects.update_or_create(
            code="SRAI-B01-PILOT",
            defaults={"name": "Book 1 Pilot", "version": "1.0.0-rc1", "status": "candidate"},
        )
        release.production_units.set(units)
        self.stdout.write(self.style.SUCCESS("SRAI Studio pilot data loaded."))
