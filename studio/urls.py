from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("curriculum/", views.curriculum, name="curriculum"),
    path("units/", views.production_units, name="production_units"),
    path("units/<int:pk>/", views.production_unit_detail, name="production_unit_detail"),
    path("work-queue/", views.work_queue, name="work_queue"),
    path("release-readiness/", views.release_readiness, name="release_readiness"),
]
