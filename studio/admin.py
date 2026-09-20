from django.contrib import admin

from .models import (
    AssetType, Book, Chapter, Notebook, ProductionAsset, ProductionTask,
    ProductionUnit, Publication, QualityReview, Release,
)

admin.site.register([
    Book, Chapter, Notebook, ProductionUnit, AssetType, ProductionAsset,
    ProductionTask, QualityReview, Publication, Release,
])
