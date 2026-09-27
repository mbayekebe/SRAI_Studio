"""Invoke the repository-contained, evidence-backed reconciliation package."""
from pathlib import Path
import subprocess
import sys
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = 'Preview/apply verified asset reconciliation and the Book 6 archive migration.'
    def add_arguments(self, parser):
        parser.add_argument('--apply', action='store_true')
        parser.add_argument('--repositories', default='C:/SRAI_GitHub')
    def handle(self, *args, **options):
        script=Path(settings.BASE_DIR)/'maintenance/studio_reconciliation_v1_1/finalize_studio.py'
        cmd=[sys.executable,str(script),'--studio',str(settings.BASE_DIR),'--repositories',options['repositories']]
        if options['apply']:cmd.append('--apply')
        if subprocess.run(cmd).returncode:raise CommandError('Reconciliation stopped. See preceding report or error.')
