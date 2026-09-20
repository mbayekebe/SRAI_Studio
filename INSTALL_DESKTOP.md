# SRAI Studio v2.0 Desktop Installation

## Requirements

- Python 3.11 or newer
- About 2 GB of free disk space for the scientific Python environment
- Internet access during the first dependency installation

## Windows

1. Extract the ZIP to a normal folder such as `Documents\\SRAI_Studio_v2`.
2. Double-click `install_windows.bat` and wait for the validation to complete.
3. Double-click `start_windows.bat`.
4. Open `http://127.0.0.1:8000/` in your browser.

To create an administrator, open Command Prompt in this folder and run:

```bat
.venv\Scripts\python manage.py createsuperuser
```

## macOS or Linux

Open Terminal in the extracted folder and run:

```bash
chmod +x install_mac_linux.sh start_mac_linux.sh
./install_mac_linux.sh
./start_mac_linux.sh
```

Then open `http://127.0.0.1:8000/`.

## What installation does

The installer creates an isolated `.venv`, installs the declared dependencies,
migrates the local SQLite database, imports all 9 books and 200 notebooks, and
runs the 11 application tests. Your source materials remain inside this folder.

MiniSEPE is not installed into the SRAI core. It is an optional, separately
identified lab extension in the complete archive.
