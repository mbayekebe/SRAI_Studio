@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv || goto :error
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements.txt || goto :error
.venv\Scripts\python manage.py migrate --noinput || goto :error
.venv\Scripts\python manage.py import_srai_catalog || goto :error
.venv\Scripts\python -m pytest -q || goto :error
echo.
echo SRAI Studio v2.0 installed and validated successfully.
echo Run start_windows.bat to launch it.
pause
exit /b 0
:error
echo.
echo Installation stopped because a command failed. Review the message above.
pause
exit /b 1
