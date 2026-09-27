# Validation

Passed using the uploaded Studio models, Django 5.2, SQLite, and a temporary database with the exact 15 legacy Book 6 chapter/notebook identities from the user's preview report:

- Legacy identity preconditions and unique archive codes.
- Archive retains original IDs, relationships and notebook validation history.
- Separate active Generative AI Book 6 with canonical Lesson 1.
- 139 asset specifications reconcile successfully; original approval times remain unchanged.
- Completion derives from registered assets; EP-M02 remains below 100% while private resources are missing.
- New Book 6 and OS-A02 core asset completion reach 100% in the fixture; no Gold Standard promotion or review creation.
- Existing reviews, publications, releases and Notebook records are preserved at field level.
- Second reconciliation is an exact database no-op.
- Mid-update failure rolls back the archive and all other data changes.
- Missing/incorrect file size/hash checks reject application.

Local Windows files are not accessible in the authoring environment. Their integrity checks run on the user's PC. PowerShell launcher/code-copy steps have not been executed on Windows here. No live website checks or new notebook executions are claimed.

The four supplied Windows files that failed v1.1 matched their original manifest hashes after pure LF/CRLF conversion. A same-size content mutation was rejected by the revised check.
