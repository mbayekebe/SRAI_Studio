@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
echo SRAI Studio is starting at http://127.0.0.1:8000/
.venv\Scripts\python manage.py runserver 127.0.0.1:8000
