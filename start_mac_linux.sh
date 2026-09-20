#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run ./install_mac_linux.sh first."
  exit 1
fi
echo "SRAI Studio is starting at http://127.0.0.1:8000/"
.venv/bin/python manage.py runserver 127.0.0.1:8000
