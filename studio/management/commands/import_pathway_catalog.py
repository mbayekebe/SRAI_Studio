from django.core.management.base import BaseCommand
from django.db import transaction

from studio.models import Pathway, ProductionUnit, Publication, QualityReview


PATHWAYS = [
    {
        "code": "EP", "title": "Executive Pathway", "slug": "executive-pathway",
        "sequence": 2, "public_url": "https://srai.mbayekebe.net/executive/",
        "description": "Decision-focused AI learning for executives and institutional leaders.",
        "modules": [
            ("EP-M01", 1, "Making a Defensible AI Decision", "0.1",
             "https://srai.mbayekebe.net/executive/modules/module-1/", ""),
            ("EP-M02", 2, "Making Trade-offs Under Risk Explicit", "0.2",
             "https://srai.mbayekebe.net/executive/modules/module-2/", "https://youtu.be/bjxj8xN9xNQ"),
            ("EP-M03", 3, "Deciding When Evidence Is Incomplete", "0.1",
             "https://srai.mbayekebe.net/executive/modules/module-3/", "https://youtu.be/jexl6cojmS0"),
        ],
    },
    {
        "code": "OS", "title": "Official Statistics & AI", "slug": "official-statistics-ai",
        "sequence": 3, "public_url": "https://srai.mbayekebe.net/official-statistics/",
        "description": "AI capability for official statisticians and national statistical systems.",
        "modules": [
            ("OS-A01", 1, "The National Information Ecosystem", "0.1.0-rc3",
             "https://srai.mbayekebe.net/official-statistics/modules/a1/", "https://youtu.be/TwsjImRahW0"),
        ],
    },
]


class Command(BaseCommand):
    help = "Import the Executive and Official Statistics pathway modules."

    @transaction.atomic
    def handle(self, *args, **options):
        imported = []
        for item in PATHWAYS:
            pathway, _ = Pathway.objects.update_or_create(
                code=item["code"],
                defaults={key: item[key] for key in (
                    "title", "slug", "sequence", "public_url", "description"
                )},
            )
            for code, number, title, version, website, video in item["modules"]:
                unit, _ = ProductionUnit.objects.update_or_create(
                    code=code,
                    defaults={
                        "chapter": None, "pathway": pathway, "module_number": number,
                        "title": title, "slug": code.lower(), "workflow_state": "published",
                        "completion_percentage": 100, "gold_standard": False,
                        "next_action": "Monitor published module and maintain evidence.",
                        "notes": f"Imported from verified SRAI Operations publication evidence; version {version}.",
                    },
                )
                Publication.objects.update_or_create(
                    production_unit=unit, platform="SRAI Website",
                    defaults={"public_url": website, "version": version, "status": "published"},
                )
                if video:
                    Publication.objects.update_or_create(
                        production_unit=unit, platform="YouTube",
                        defaults={"public_url": video, "version": version, "status": "published"},
                    )
                QualityReview.objects.update_or_create(
                    production_unit=unit, asset=None, review_type="publication",
                    defaults={"score": 100, "passed": True, "findings": "Verified publication acceptance evidence."},
                )
                imported.append(code)
        self.stdout.write(self.style.SUCCESS(
            f"Imported {len(imported)} pathway modules: {', '.join(imported)}"
        ))
