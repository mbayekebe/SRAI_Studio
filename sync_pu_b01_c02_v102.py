from datetime import date

from django.db import transaction
from django.utils import timezone

from studio.models import ProductionAsset, ProductionTask, ProductionUnit, Publication, Release


UNIT_CODE = "PU-B01-C02"
EXECUTIVE_BRIEF_URL = (
    "https://github.com/mbayekebe/srai-book-01-mathematical-foundations/"
    "blob/pu-b01-c02-v1.0.2/production-units/PU-B01-C02/docs/"
    "PU-B01-C02_Executive_Brief_v1.0.pdf"
)
EXECUTIVE_BRIEF_PATH = (
    "C:\\SRAI_GitHub\\SRAI_Book_01_Mathematical_Foundations\\production-units\\"
    "PU-B01-C02\\docs\\PU-B01-C02_Executive_Brief_v1.0.pdf"
)


with transaction.atomic():
    unit = ProductionUnit.objects.select_for_update().get(code=UNIT_CODE)
    brief = ProductionAsset.objects.select_for_update().get(
        production_unit=unit,
        asset_type__code="executive_brief",
    )

    brief.title = "PU-B01-C02 Executive Brief"
    brief.status = "published"
    brief.version = "1.0"
    brief.file_path = EXECUTIVE_BRIEF_PATH
    brief.public_url = EXECUTIVE_BRIEF_URL
    brief.notes = (
        "Controlled Executive Brief published with PU-B01-C02 v1.0.2 on "
        "2026-08-27. PDF and DOCX formats are included in the tagged release."
    )
    brief.save()

    publication = unit.publications.filter(
        asset=brief,
        platform__iexact="Github",
    ).first()
    if publication is None:
        publication = Publication(production_unit=unit, asset=brief)
    publication.platform = "Github"
    publication.public_url = EXECUTIVE_BRIEF_URL
    publication.version = "1.0"
    publication.published_at = timezone.now()
    publication.status = "published"
    publication.save()

    release = Release.objects.filter(code="REL-PU-B01-C02-V1.0.2").first()
    if release is None:
        release = Release.objects.filter(code="REL-PU-B01-C02-V1.0.1").first()
    if release is None:
        release = Release()
    release.code = "REL-PU-B01-C02-V1.0.2"
    release.name = "PU-B01-C02 Controlled Patch Release 1.0.2"
    release.version = "1.0.2"
    release.status = "candidate"
    release.release_date = date(2026, 8, 27)
    release.notes = (
        "GitHub platform release pu-b01-c02-v1.0.2 was published on 2026-08-27 "
        "with the controlled Executive Brief. The SRAI Studio release remains "
        "Candidate pending LinkedIn publication and formal quality reviews."
    )
    release.save()
    release.production_units.add(unit)

    task = unit.tasks.filter(status__in=("open", "in_progress")).order_by("id").first()
    if task is None:
        task = ProductionTask(production_unit=unit)
    task.title = "Publish Lesson 2 LinkedIn announcement and complete quality reviews"
    task.description = (
        "Record the final LinkedIn publication URL, then complete and record the "
        "technical, educational, editorial and publication reviews."
    )
    task.status = "in_progress"
    task.priority = "high"
    task.save()

    unit.workflow_state = "media_production"
    unit.next_action = "Publish LinkedIn announcement and record formal quality reviews"
    unit.notes = (
        "PU-B01-C02 v1.0.2 and its Executive Brief are published on GitHub. "
        "LinkedIn publication and formal quality reviews remain outstanding."
    )
    unit.save(update_fields=["workflow_state", "next_action", "notes"])
    unit.recalculate_completion()

    unit.refresh_from_db()
    release.refresh_from_db()
    print(
        "UPDATED",
        unit.code,
        unit.workflow_state,
        float(unit.completion_percentage),
        unit.gold_standard,
        unit.release_ready,
    )
    print("EXECUTIVE_BRIEF", brief.status, brief.version, brief.public_url)
    print("PUBLICATIONS", unit.publications.count())
    print("RELEASE", release.code, release.version, release.status, "BLOCKERS", release.blockers)
