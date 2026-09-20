from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404, render
from django.utils import timezone

from .models import Book, ProductionTask, ProductionUnit, Release


@login_required
def dashboard(request):
    units = ProductionUnit.objects.select_related("chapter__book")
    context = {
        "total_units": units.count(),
        "gold_units": units.filter(gold_standard=True).count(),
        "units_in_review": units.filter(workflow_state__in=["educational_review", "quality_review"]).count(),
        "overdue_tasks": ProductionTask.objects.exclude(status="done").filter(due_date__lt=timezone.localdate()).count(),
        "release_ready_units": sum(1 for unit in units.prefetch_related("assets__asset_type", "reviews") if unit.release_ready),
        "average_completion": units.aggregate(value=Avg("completion_percentage"))["value"] or 0,
        "workflow_counts": units.values("workflow_state").annotate(total=Count("id")).order_by("workflow_state"),
        "books": Book.objects.annotate(unit_count=Count("chapters__production_unit")).order_by("sequence"),
    }
    return render(request, "studio/dashboard.html", context)


@login_required
def curriculum(request):
    books = Book.objects.prefetch_related("chapters__notebook", "chapters__production_unit")
    return render(request, "studio/curriculum.html", {"books": books})


@login_required
def production_units(request):
    units = ProductionUnit.objects.select_related("chapter__book", "owner")
    state = request.GET.get("state")
    if state:
        units = units.filter(workflow_state=state)
    return render(request, "studio/production_units.html", {"units": units, "states": ProductionUnit.STATES})


@login_required
def production_unit_detail(request, pk):
    unit = get_object_or_404(
        ProductionUnit.objects.select_related("chapter__book", "owner").prefetch_related(
            "assets__asset_type", "tasks", "reviews", "publications"
        ),
        pk=pk,
    )
    return render(request, "studio/production_unit_detail.html", {"unit": unit})


@login_required
def work_queue(request):
    tasks = ProductionTask.objects.exclude(status="done").select_related("production_unit", "assignee").order_by(
        "due_date", "-priority"
    )
    if not request.user.is_staff:
        tasks = tasks.filter(Q(assignee=request.user) | Q(assignee__isnull=True))
    return render(request, "studio/work_queue.html", {"tasks": tasks})


@login_required
def release_readiness(request):
    releases = Release.objects.prefetch_related("production_units__assets__asset_type", "production_units__reviews")
    return render(request, "studio/release_readiness.html", {"releases": releases})
