from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class Book(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    edition = models.CharField(max_length=50, default="1.0")
    status = models.CharField(max_length=30, default="active")
    sequence = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ["sequence", "code"]

    def __str__(self):
        return f"{self.code} — {self.title}"


class Pathway(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=30, default="active")
    sequence = models.PositiveIntegerField(default=1)
    public_url = models.URLField(blank=True)

    class Meta:
        ordering = ["sequence", "code"]

    def __str__(self):
        return f"{self.code} — {self.title}"


class Chapter(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="chapters")
    code = models.CharField(max_length=30, unique=True)
    number = models.PositiveIntegerField()
    title = models.CharField(max_length=255)
    slug = models.SlugField()
    summary = models.TextField(blank=True)
    status = models.CharField(max_length=30, default="planned")
    estimated_study_minutes = models.PositiveIntegerField(default=60)

    class Meta:
        ordering = ["book__sequence", "number"]
        constraints = [
            models.UniqueConstraint(fields=["book", "number"], name="unique_book_chapter_number"),
            models.UniqueConstraint(fields=["book", "slug"], name="unique_book_chapter_slug"),
        ]

    def __str__(self):
        return f"{self.code} — {self.title}"


class Notebook(models.Model):
    VALIDATION = [("not_tested", "Not tested"), ("passed", "Passed"), ("failed", "Failed")]
    chapter = models.OneToOneField(Chapter, on_delete=models.CASCADE, related_name="notebook")
    code = models.CharField(max_length=30, unique=True)
    title = models.CharField(max_length=255)
    repository_path = models.CharField(max_length=500)
    package_name = models.CharField(max_length=100, blank=True)
    validation_status = models.CharField(max_length=20, choices=VALIDATION, default="not_tested")
    last_validated_at = models.DateTimeField(null=True, blank=True)
    validation_message = models.TextField(blank=True)
    execution_seconds = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)

    def __str__(self):
        return self.code


class ProductionUnit(models.Model):
    STATES = [
        ("planned", "Planned"), ("drafting", "Drafting"),
        ("educational_review", "Educational Review"), ("media_production", "Media Production"),
        ("quality_review", "Quality Review"), ("ready", "Ready for Publication"),
        ("published", "Published"), ("revised", "Revised"),
    ]
    PRIORITIES = [("low", "Low"), ("normal", "Normal"), ("high", "High"), ("urgent", "Urgent")]
    chapter = models.OneToOneField(
        Chapter, null=True, blank=True, on_delete=models.CASCADE, related_name="production_unit"
    )
    pathway = models.ForeignKey(
        Pathway, null=True, blank=True, on_delete=models.CASCADE, related_name="modules"
    )
    module_number = models.PositiveIntegerField(null=True, blank=True)
    title = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(blank=True)
    summary = models.TextField(blank=True)
    estimated_study_minutes = models.PositiveIntegerField(default=60)
    code = models.CharField(max_length=40, unique=True)
    workflow_state = models.CharField(max_length=30, choices=STATES, default="planned")
    priority = models.CharField(max_length=20, choices=PRIORITIES, default="normal")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    due_date = models.DateField(null=True, blank=True)
    completion_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    gold_standard = models.BooleanField(default=False)
    next_action = models.CharField(max_length=255, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(chapter__isnull=False, pathway__isnull=True)
                    | models.Q(chapter__isnull=True, pathway__isnull=False)
                ),
                name="unit_has_exactly_one_parent",
            ),
            models.UniqueConstraint(
                fields=["pathway", "module_number"],
                condition=models.Q(pathway__isnull=False),
                name="unique_pathway_module_number",
            ),
        ]

    def recalculate_completion(self, save=True):
        assets = self.assets.select_related("asset_type")
        denominator = sum(float(a.asset_type.weight) for a in assets if a.asset_type.required)
        numerator = sum(
            float(a.asset_type.weight)
            for a in assets
            if a.asset_type.required and a.status in ProductionAsset.COMPLETE_STATUSES
        )
        self.completion_percentage = round((numerator / denominator * 100) if denominator else 0, 2)
        self.gold_standard = denominator > 0 and numerator == denominator
        if save:
            self.save(update_fields=["completion_percentage", "gold_standard"])
        return self.completion_percentage

    @property
    def display_title(self):
        return self.chapter.title if self.chapter_id else self.title

    @property
    def programme_family(self):
        return self.pathway.title if self.pathway_id else "Professional Programme"

    @property
    def parent_code(self):
        return self.pathway.code if self.pathway_id else self.chapter.book.code

    @property
    def release_ready(self):
        required_assets = self.assets.filter(asset_type__required=True)
        assets_ready = required_assets.exists() and not required_assets.exclude(
            status__in=ProductionAsset.COMPLETE_STATUSES
        ).exists()
        reviews_ready = self.reviews.exists() and not self.reviews.filter(passed=False).exists()
        return assets_ready and reviews_ready

    def __str__(self):
        return self.code


class AssetType(models.Model):
    FAMILIES = [("educational", "Educational"), ("communication", "Communication"), ("professional", "Professional")]
    code = models.CharField(max_length=40, unique=True)
    name = models.CharField(max_length=150)
    family = models.CharField(max_length=20, choices=FAMILIES)
    required = models.BooleanField(default=True)
    weight = models.DecimalField(max_digits=6, decimal_places=2, default=1)
    sequence = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ["family", "sequence", "code"]

    def __str__(self):
        return self.name


class ProductionAsset(models.Model):
    COMPLETE_STATUSES = ("approved", "published", "not_applicable")
    STATUSES = [
        ("not_started", "Not Started"), ("drafting", "Drafting"), ("review", "In Review"),
        ("approved", "Approved"), ("published", "Published"), ("not_applicable", "Not Applicable"),
    ]
    production_unit = models.ForeignKey(ProductionUnit, on_delete=models.CASCADE, related_name="assets")
    asset_type = models.ForeignKey(AssetType, on_delete=models.PROTECT)
    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUSES, default="not_started")
    version = models.CharField(max_length=30, default="0.1")
    file_path = models.CharField(max_length=500, blank=True)
    public_url = models.URLField(blank=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    due_date = models.DateField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["production_unit", "asset_type"], name="unique_asset_per_unit")]

    def save(self, *args, **kwargs):
        if self.status in self.COMPLETE_STATUSES and not self.approved_at:
            self.approved_at = timezone.now()
        super().save(*args, **kwargs)
        self.production_unit.recalculate_completion()

    def __str__(self):
        return f"{self.production_unit.code}: {self.asset_type.name}"


class ProductionTask(models.Model):
    STATUSES = [("open", "Open"), ("in_progress", "In Progress"), ("blocked", "Blocked"), ("done", "Done")]
    production_unit = models.ForeignKey(ProductionUnit, on_delete=models.CASCADE, related_name="tasks")
    asset = models.ForeignKey(ProductionAsset, null=True, blank=True, on_delete=models.CASCADE, related_name="tasks")
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUSES, default="open")
    priority = models.CharField(max_length=20, choices=ProductionUnit.PRIORITIES, default="normal")
    assignee = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    due_date = models.DateField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if self.status == "done" and not self.completed_at:
            self.completed_at = timezone.now()
        super().save(*args, **kwargs)


class QualityReview(models.Model):
    TYPES = [("technical", "Technical"), ("educational", "Educational"), ("editorial", "Editorial"), ("publication", "Publication")]
    production_unit = models.ForeignKey(ProductionUnit, on_delete=models.CASCADE, related_name="reviews")
    asset = models.ForeignKey(ProductionAsset, null=True, blank=True, on_delete=models.CASCADE, related_name="reviews")
    review_type = models.CharField(max_length=20, choices=TYPES)
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    score = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    passed = models.BooleanField(default=False)
    findings = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(default=timezone.now)


class Publication(models.Model):
    STATUSES = [("planned", "Planned"), ("scheduled", "Scheduled"), ("published", "Published"), ("withdrawn", "Withdrawn")]
    production_unit = models.ForeignKey(ProductionUnit, on_delete=models.CASCADE, related_name="publications")
    asset = models.ForeignKey(ProductionAsset, null=True, blank=True, on_delete=models.SET_NULL, related_name="publications")
    platform = models.CharField(max_length=100)
    public_url = models.URLField(blank=True)
    version = models.CharField(max_length=30, default="1.0")
    published_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUSES, default="planned")


class Release(models.Model):
    STATUSES = [("planned", "Planned"), ("candidate", "Candidate"), ("released", "Released"), ("retired", "Retired")]
    code = models.CharField(max_length=40, unique=True)
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=30)
    status = models.CharField(max_length=20, choices=STATUSES, default="planned")
    production_units = models.ManyToManyField(ProductionUnit, related_name="releases")
    release_date = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)

    @property
    def blockers(self):
        return [unit.code for unit in self.production_units.all() if not unit.release_ready]
