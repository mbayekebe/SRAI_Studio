import json
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from studio.models import Notebook


class Command(BaseCommand):
    help = "Apply a notebook-execution JSON report to SRAI Studio."

    def add_arguments(self, parser):
        parser.add_argument("report")

    def handle(self, *args, **options):
        path = Path(options["report"])
        if not path.is_file():
            raise CommandError(f"Report does not exist: {path}")
        payload = json.loads(path.read_text())
        updated = 0
        missing = []
        for row in payload["results"]:
            try:
                notebook = Notebook.objects.get(code=row["code"])
            except Notebook.DoesNotExist:
                missing.append(row["code"])
                continue
            notebook.validation_status = "passed" if row["status"] == "passed" else "failed"
            notebook.validation_message = row.get("error", "")
            notebook.execution_seconds = Decimal(str(row.get("duration_seconds", 0)))
            notebook.last_validated_at = timezone.now()
            notebook.save(
                update_fields=[
                    "validation_status", "validation_message",
                    "execution_seconds", "last_validated_at",
                ]
            )
            updated += 1
        if missing:
            raise CommandError(f"Unknown notebook codes: {missing}")
        self.stdout.write(self.style.SUCCESS(f"Updated validation status for {updated} notebooks."))
