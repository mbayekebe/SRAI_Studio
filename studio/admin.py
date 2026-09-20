from django.contrib import admin

from .models import (
    AssetType, Book, Chapter, Notebook, ProductionAsset, ProductionTask,
    Pathway, ProductionUnit, Publication, QualityReview, Release,
)

admin.site.register([
    Book, Pathway, Chapter, Notebook, ProductionUnit, AssetType, ProductionAsset,
    ProductionTask, QualityReview, Publication, Release,
])
