# SRAI Studio Version Reconciliation Report v1.0

**Project:** Statistics, Reasoning and Artificial Intelligence (SRAI)  
**Report date:** 21 August 2026  
**Owner/author:** Mbaye Kebe  
**Scope:** SRAI Studio packages from MVP v0.1.0 through v1.6.0, including working-edition and consolidated-package variants.

## 1. Executive decision

The SRAI Studio history is valid and recoverable, but **no existing package is fully aligned with the current nine-book, 200-chapter, 200-notebook SRAI baseline**.

The controlling application decisions are:

1. **v1.0 Working Edition** is the first verified runnable Professional baseline.
2. **v1.4.0 Fully Validated** is the strongest technically validated eight-book/185-notebook baseline.
3. **v1.5.0 Teaching and Publication RC** is the latest SRAI-only core release candidate and therefore the controlling source baseline for the next application release.
4. **v1.6.0 MiniSEPE Integrated** is an optional applied-laboratory extension. It is not the canonical SRAI core.
5. `SRAI_04_SRAI_Studio_v1.0.zip` is a consolidated derivative package of the v1.6 integrated state, despite its ambiguous filename.
6. The next canonical application must be a **SRAI Studio v2.0 Professional Edition** upgraded to Books 1–9 and the validated 200-notebook corpus.

## 2. Product boundary

SRAI Studio is an internal Django production-management platform for:

- books, chapters, and notebooks;
- Production Units;
- educational, communication, and professional assets;
- work queues, tasks, milestones, and calendars;
- workflow transitions and quality reviews;
- publication and release readiness;
- roles, permissions, notifications, and audit history;
- controlled curriculum import and notebook validation evidence.

It is not, by default, a public learning-management system, payment platform, public website, marketing-automation platform, or MiniSEPE production deployment.

## 3. Recovered version progression

| Version/package | Principal scope or delta | Evidence status | Disposition |
|---|---|---|---|
| v0.1.0 | Initial MVP; Book 1 Chapters 1–5 pilot; core registry, Production Units, assets, tasks, QA/publication models, dashboard skeleton, Docker | ZIP valid | Historical MVP |
| v0.2.0 | Editable assets, task queue, workflow transitions, quality reviews, readiness, search/filtering, expanded dashboard | ZIP valid; feature manifest present | Historical MVP |
| v0.3.0 | Roles/capabilities, audit history, release creation, charts, CSV bulk import, nine-volume template | ZIP valid; feature manifest present | Historical MVP |
| v0.4.0 | HTMX inline updates, file references, milestones, calendar, notifications, JSON API, migrations | ZIP valid; feature manifest present | Historical MVP |
| v0.5.0 | Managed uploads, Docker media, email delivery, health endpoint, security hardening, 200-unit import template | ZIP valid | Pre-professional milestone; explicitly not final |
| MVP v1.0.0 | Compact pilot-oriented Django package; still describes Book 1 Chapters 1–5 and defers full logical-notebook import | ZIP valid | Superseded pilot package |
| v1.0 Working Edition | First runnable, migrated, tested, seeded, rendered, and server-verified Django edition; nine automated tests passed | Two packages recovered; one includes deployment/static artifacts | First Professional baseline |
| v1.1.0 Full Catalog | Imports eight books, 185 chapters/notebooks/Production Units; two historical M1-N18 variants archived | ZIP valid | Superseded catalog baseline |
| v1.2.0 Execution Audit | Adds deterministic execution audit; initially reports 0/185 because eight required `srai_*` packages were absent | ZIP valid; failure accurately recorded | Diagnostic baseline |
| v1.3.0 Package Recovery | Recovers required SRAI scientific-support packages | ZIP valid | Remediation baseline |
| v1.4.0 Fully Validated | 185/185 notebooks passed, 0 failed; 11 Studio tests passed; eight support packages operational | ZIP valid; recorded SHA-256 matches recovered package | Technical eight-book baseline |
| v1.5.0 Teaching Publication RC | Adds teaching/publication release kit, pathways, instructor guidance, assessment criteria, outreach, catalog export, release gates | ZIP valid; release candidate; editorial/licensing/pilot/public-infrastructure gates pending | **Canonical SRAI-only source baseline** |
| v1.6.0 MiniSEPE Integrated | Adds 30 MiniSEPE guided labs, reference package, validator, and completion report | ZIP valid; core files otherwise unchanged from v1.5 except documentation/manifest | Optional applied-lab extension |
| Package 04 | Wrapper/consolidated export containing the v1.6 application state and 30 MiniSEPE labs | ZIP valid; ambiguous `v1.0` filename but embedded manifest says v1.6.0 | Archive as derivative; do not use as canonical version label |

## 4. v1.5 versus v1.6 code boundary

A recursive comparison shows that v1.6 changes are bounded to:

- root README changes;
- new `labs/` directory;
- release README and release-manifest changes;
- new `TRACK_B_COMPLETION_REPORT.md`;
- new `validate_minisepe_lab.py`.

The core Django application remains unchanged. This confirms that MiniSEPE integration can be separated cleanly as an optional extension without forking the SRAI core.

## 5. Canonical source baseline

The controlling code ancestor for the next release is:

`SRAI_Studio_v1.5.0_Teaching_Publication_RC.zip`

Recovered SHA-256:

`923815100bf8a5b59ae030ba7cf7b62cda225213e9d10f947fb6104595693671`

Reasons:

- it contains the latest SRAI-only application core;
- it includes the teaching/publication release kit;
- it excludes the optional MiniSEPE laboratory;
- its core is the same application code carried into v1.6;
- it retains honest pending release gates rather than implying public authorization.

## 6. Why v1.6 is not the canonical core

v1.6 correctly states that MiniSEPE is separate from the canonical SRAI catalog and that its 30 labs are not a ninth volume. Nevertheless, the package combines an application release with a domain-specific applied laboratory. For project governance and reuse:

- SRAI core must remain domain-neutral;
- MiniSEPE must be installable or mountable as an optional lab extension;
- MiniSEPE labs must not alter canonical book, chapter, or notebook counts;
- live SEPE data, credentials, and confidential configuration must remain excluded.

## 7. Current baseline gap

All v1.1–v1.6 full-catalog releases are based on:

- 8 books;
- 185 chapters;
- 185 active notebooks;
- 2 archived M1-N18 historical variants.

The current SRAI canonical baseline is:

- 9 books;
- 200 chapters;
- 200 canonical notebooks;
- Book 9: AI Transformation and Executive Leadership.

Therefore, v1.5 and v1.6 are historically valid but structurally outdated for the current series.

## 8. Required v2.0 Professional Edition changes

1. Replace the eight-book/185-row import catalog with the authoritative Stage 67 200-row notebook map.
2. Add Book 9, its 15 chapters, notebook records, Production Units, and Gold Standard asset checklists.
3. Integrate the validated 200-notebook release produced by the consolidation project.
4. Update all hard-coded `185` and eight-volume guards, reports, dashboards, tests, and release manifests.
5. Preserve the two historical M1-N18 variants in the archive, excluded from active counts.
6. Correct the SRAI expansion everywhere to “Statistics, Reasoning and Artificial Intelligence.”
7. Keep MiniSEPE outside the core package as an optional `labs/minisepe` extension with its own manifest and version.
8. Rerun Django migrations, system checks, Studio tests, 200-notebook execution, browser smoke tests, and release-readiness checks.
9. Keep editorial review, rights/licensing, pilot delivery, and public-infrastructure authorization as explicit gates.

## 9. Package disposition

| Category | Packages |
|---|---|
| Canonical source baseline | v1.5.0 Teaching Publication RC |
| Technical reference baseline | v1.4.0 Fully Validated |
| First Professional/runnable baseline | v1.0 Working Edition |
| Optional extension reference | v1.6.0 MiniSEPE Integrated |
| Historical development archive | v0.1–v0.5, MVP v1.0.0, v1.1–v1.3 |
| Derivative/ambiguous wrapper | `SRAI_04_SRAI_Studio_v1.0.zip` |

## 10. Next controlled operation

Build **SRAI Studio v2.0 Professional Edition** from the v1.5 SRAI-only core, migrate it to the nine-book/200-notebook catalog, validate it, and package MiniSEPE separately as an optional extension.
