# SRAI Studio final reconciliation v1.1

This package finishes reconciliation of the verified asset inventory supplied on 27 September 2026. It does not claim that missing future curriculum or private teaching files have been supplied. No website deployment is involved.

## Expected changes from the applied v1.0 baseline

- Archive the original 15 Decision Intelligence chapters under Book `B06-LEGACY` (status `archived`), with unit/chapter codes `LEGACY-B06-C01` through `LEGACY-B06-C15`. Primary keys, relations, notebook files, validation history, assets, reviews, publications, and releases remain intact. The original book slug is preserved.
- Create active Book `B06`, Generative Artificial Intelligence, and verified Lesson 1 / notebook `M6_N01`. New notebook PASS records the supplied owner-confirmed execution evidence and matching file checksum; the updater does not execute the notebook.
- Map 139 asset records across 15 units from canonical manifests and publication evidence. This covers the newly published technical lessons, current Book 6 Lesson 1, and six pathway modules. Book 1 Lessons 1â€“5 and Book 7 asset records remain unchanged.
- Verify 92 local files by existence and recorded size, plus checksum where the supplied manifest provides an unambiguous SHA-256. Text files are also accepted when a pure LF/CRLF conversion reproduces the manifest hash; the report records which conversion was needed. Record all computed hashes.
- Calculate completion from registered required assets; preserve existing Gold Standard flags and review records. No approval timestamps or review scores are fabricated.
- Expected count from the 256-unit baseline: 257 units total, including the 15 archived units. Expected books: nine active and one archived. Existing UI totals may include the archive; this package does not alter templates.
- Install a self-contained maintenance command and its evidence within Studio so the repair is reproducible from Git. Runtime reports/backups remain excluded from Git.

## Run preview

Extract this folder under `C:\SRAI_GitHub`. Stop Studio and other database writers. Preview simulates the changes inside a transaction and rolls it back; it is not an SQL read-only connection.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\SRAI_GitHub\SRAI_Studio_Final_Update_v1.1.1\Run_Final_Update.ps1"
```

Return the printed result. If a local file check fails, the report identifies the exact file and check. Do not bypass a failure by editing the expected hash or size.

## Apply

After the preview checks succeed, with Studio still stopped:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\SRAI_GitHub\SRAI_Studio_Final_Update_v1.1.1\Run_Final_Update.ps1" -Apply
```

Apply creates an integrity-checked SQLite backup before data changes. All database changes are one transaction. It verifies original notebook/review/publication/release history, approval timestamps, Gold Standard flags, and repeat-run idempotence. Before-state records and the backup path are saved in `results/<timestamp>/report.json`.

Code installation follows the database transaction. If installation fails after database success, retain the report and rerun the installer; applying the database operation again is idempotent. Existing custom code at `reconcile_studio_records.py` is backed up before replacement. This package does not commit or push.

After installation the maintenance command is:

```powershell
& '.\.venv\Scripts\python.exe' manage.py reconcile_studio_records
```

It previews by default; use `--apply` for application. `--repositories` can select another repository root.

Use this command for the current maintenance workflow. The older generic `import_srai_catalog` uses the original curriculum and can reset validation data. The v1.0 `import_pathway_catalog` command remains installed with its explicit external-package requirement; it is not the current asset/curriculum maintenance command.

## Boundaries and honest outstanding work

Two blocked production tasks remain visible:

1. EP-M02 private controlled-resource inventory: the supplied repository inventory contains no relevant teaching files. Its page and video are recorded, private downloads remain disabled, and a required incomplete resource record prevents an unsupported 100% completion claim.
2. Approved future Generative AI Book 6 syllabus: the supplied repository contains Lesson 1 only. The remaining planned lesson titles and assets are not invented. The old Decision Intelligence lessons are retained as an archive; they are not equated one-to-one with revised Executive modules.

EP-M01's newer manifest names a *release target* rather than proving a newer public release. Its local files are recorded as approved, not newly published, and no newer release publication is invented. EP-M03 and OS asset publication states use supplied acceptance records. OS-A02 restricted exclusions and OS-A03 owner-authorized public solutions remain distinct.

Existing Notebook rows remain historical baseline records. Current published notebook paths are mapped in ProductionAsset; replacing a baseline Notebook row could incorrectly transfer its validation history to a different file. New Book 6 receives its own Notebook row. Existing synthetic pathway publication reviews are retained for traceability, not treated as fresh independent review.

## Recovery

Stop Studio and every SQLite writer. Preserve the post-update database and any WAL/SHM files separately before restoring `Studio_before.sqlite3` to the configured database path. Restore code from the relevant backup or reviewed Git baseline. Never restore over a running SQLite application. The report contains before-state fields and backup checksum.

