import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("studio", "0002_notebook_execution_seconds_and_more")]

    operations = [
        migrations.CreateModel(
            name="Pathway",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("code", models.CharField(max_length=20, unique=True)),
                ("title", models.CharField(max_length=255)),
                ("slug", models.SlugField(unique=True)),
                ("description", models.TextField(blank=True)),
                ("status", models.CharField(default="active", max_length=30)),
                ("sequence", models.PositiveIntegerField(default=1)),
                ("public_url", models.URLField(blank=True)),
            ],
            options={"ordering": ["sequence", "code"]},
        ),
        migrations.AlterField(
            model_name="productionunit", name="chapter",
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="production_unit", to="studio.chapter"),
        ),
        migrations.AddField(model_name="productionunit", name="estimated_study_minutes", field=models.PositiveIntegerField(default=60)),
        migrations.AddField(model_name="productionunit", name="module_number", field=models.PositiveIntegerField(blank=True, null=True)),
        migrations.AddField(model_name="productionunit", name="slug", field=models.SlugField(blank=True)),
        migrations.AddField(model_name="productionunit", name="summary", field=models.TextField(blank=True)),
        migrations.AddField(model_name="productionunit", name="title", field=models.CharField(blank=True, max_length=255)),
        migrations.AddField(
            model_name="productionunit", name="pathway",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="modules", to="studio.pathway"),
        ),
        migrations.AddConstraint(
            model_name="productionunit",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(chapter__isnull=False, pathway__isnull=True)
                    | models.Q(chapter__isnull=True, pathway__isnull=False)
                ),
                name="unit_has_exactly_one_parent",
            ),
        ),
        migrations.AddConstraint(
            model_name="productionunit",
            constraint=models.UniqueConstraint(
                fields=("pathway", "module_number"),
                condition=models.Q(pathway__isnull=False),
                name="unique_pathway_module_number",
            ),
        ),
    ]
