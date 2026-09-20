import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.urls import reverse

from .models import Book, Pathway, ProductionAsset, ProductionUnit, Publication, Release


@pytest.fixture
def seeded(db):
    call_command("seed_pilot")


@pytest.fixture
def client_logged_in(client, db):
    user = get_user_model().objects.create_user("director", password="safe-test-password", is_staff=True)
    client.force_login(user)
    return client


@pytest.mark.django_db
def test_seed_is_idempotent():
    call_command("seed_pilot")
    call_command("seed_pilot")
    assert Book.objects.count() == 1
    assert ProductionUnit.objects.count() == 5
    assert ProductionAsset.objects.count() == 35


@pytest.mark.django_db
def test_gold_standard_completion(seeded):
    gold = ProductionUnit.objects.get(code="PU-B01-C01")
    assert gold.completion_percentage == 100
    assert gold.gold_standard is True
    assert gold.release_ready is True


@pytest.mark.django_db
def test_completion_updates_when_asset_status_changes(seeded):
    unit = ProductionUnit.objects.get(code="PU-B01-C02")
    asset = unit.assets.filter(status="not_started").first()
    before = float(unit.completion_percentage)
    asset.status = "approved"
    asset.save()
    unit.refresh_from_db()
    assert float(unit.completion_percentage) > before


@pytest.mark.django_db
def test_release_exposes_blockers(seeded):
    release = Release.objects.get(code="SRAI-B01-PILOT")
    assert "PU-B01-C01" not in release.blockers
    assert "PU-B01-C02" in release.blockers


@pytest.mark.django_db
@pytest.mark.parametrize("name", ["dashboard", "curriculum", "production_units", "work_queue", "release_readiness"])
def test_secure_pages_render(client_logged_in, seeded, name):
    response = client_logged_in.get(reverse(name))
    assert response.status_code == 200


@pytest.mark.django_db
def test_anonymous_user_is_redirected(client):
    response = client.get(reverse("dashboard"))
    assert response.status_code == 302
    assert "/accounts/login/" in response.url


@pytest.mark.django_db
def test_full_catalog_import(settings):
    call_command("import_srai_catalog", root=settings.BASE_DIR / "content" / "notebooks")
    from .models import Chapter, Notebook
    assert Book.objects.count() == 9
    assert Chapter.objects.count() == 250
    assert Notebook.objects.count() == 250
    assert ProductionUnit.objects.count() == 250
    assert ProductionAsset.objects.count() == 250 * 7
    assert Book.objects.get(code="B09").title == "AI Transformation and Executive Leadership"


@pytest.mark.django_db
def test_pathway_catalog_import_is_idempotent():
    call_command("import_pathway_catalog")
    call_command("import_pathway_catalog")
    assert Pathway.objects.count() == 2
    assert ProductionUnit.objects.filter(pathway__isnull=False).count() == 4
    assert set(ProductionUnit.objects.filter(pathway__isnull=False).values_list("code", flat=True)) == {
        "EP-M01", "EP-M02", "EP-M03", "OS-A01"
    }
    assert Publication.objects.filter(production_unit__pathway__isnull=False).count() == 7


@pytest.mark.django_db
def test_pathway_pages_and_filters(client_logged_in):
    call_command("import_pathway_catalog")
    response = client_logged_in.get(reverse("dashboard"))
    assert response.status_code == 200
    assert b"Official Statistics &amp; AI" in response.content
    response = client_logged_in.get(reverse("production_units"), {"family": "OS"})
    assert b"OS-A01" in response.content
    assert b"EP-M01" not in response.content
    unit = ProductionUnit.objects.get(code="EP-M01")
    response = client_logged_in.get(reverse("production_unit_detail", args=[unit.pk]))
    assert response.status_code == 200
    assert b"Making a Defensible AI Decision" in response.content
