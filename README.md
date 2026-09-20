# SRAI Studio v2.0 Professional Edition

Internal Django production-management platform for the **Statistics, Reasoning and Artificial Intelligence (SRAI)** educational ecosystem.

## Included

- Secure authentication and Django administration
- Curriculum registry for books, chapters, and notebooks
- Production Units and weighted Gold Standard asset tracking
- Work queue, quality reviews, publications, and release readiness
- Executive dashboard
- Idempotent Book 1 Chapters 1–5 pilot seed
- SQLite quick start and PostgreSQL 16 Docker Compose deployment
- Automated acceptance tests
- Complete nine-book, 200-notebook canonical catalog
- Book 9: AI Transformation and Executive Leadership

## Local validation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py seed_pilot
.venv/bin/python manage.py import_srai_catalog
.venv/bin/pytest
```

Create an administrator and start:

```bash
.venv/bin/python manage.py createsuperuser
.venv/bin/python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Docker

```bash
cp .env.example .env
docker compose up --build
docker compose exec web python manage.py createsuperuser
```

The platform manages the complete nine-book, 200-notebook catalog:

```bash
.venv/bin/python manage.py import_srai_catalog
```

This creates nine books, 200 chapters, 200 notebook records, 200 Production
Units, and their Gold Standard asset checklists. Historical variants remain
outside the active catalog and are preserved in the wider SRAI archive.

## Notebook execution audit

```bash
python scripts/execute_notebooks.py --report reports/final_execution_report.json
python manage.py apply_validation_report reports/final_execution_report.json
```

The executor requires the nine `srai_*` scientific support packages in the
Python environment. A missing support package is recorded as a failed
validation; it is never treated as a successful notebook run.

## Publication and teaching release

The v2.0 release kit is under `release/`. It contains curriculum
pathways, instructor guidance, assessment criteria, publication gates, outreach
copy, a machine-readable release manifest, and the exported notebook catalog.

Generate the catalog from the current Studio database with:

```bash
python scripts/export_release_catalog.py
```

## Product boundary

SRAI Studio core is domain-neutral. MiniSEPE is maintained as a separate,
optional applied-laboratory extension and is not bundled into this core release.

For desktop installation, see `INSTALL_DESKTOP.md` or use the included Windows
and macOS/Linux helper scripts.
