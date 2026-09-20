#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


PATTERN = re.compile(r"^((?:M[1-4]|V[5689]|V7_A)_N\d{2})_")


def execute_inprocess(path):
    match = PATTERN.match(path.name)
    code = match.group(1) if match else path.stem
    started = time.monotonic()
    namespace = {"__name__": "__main__"}
    index = -1
    source = ""
    try:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source.strip()
            if not source:
                continue
            # The published notebooks use /mnt/data as their portable artifact
            # directory. Map it to a writable runtime location in restricted CI.
            source = source.replace('"/mnt/data"', '"/tmp/srai-notebook-runtime/data"')
            source = source.replace("'/mnt/data'", "'/tmp/srai-notebook-runtime/data'")
            compiled = compile(source, f"{path.name}:cell-{index}", "exec")
            exec(compiled, namespace, namespace)
        return {
            "code": code, "file": str(path), "status": "passed",
            "duration_seconds": round(time.monotonic() - started, 3), "error": "",
        }
    except Exception as exc:
        return {
            "code": code, "file": str(path), "status": "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "cell_index": index,
            "cell_source": source[:4000],
            "traceback": traceback.format_exc(),
        }


def execute_kernel(path, timeout):
    match = PATTERN.match(path.name)
    code = match.group(1) if match else path.stem
    started = time.monotonic()
    try:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute(cwd=str(path.parent))
        return {
            "code": code,
            "file": str(path),
            "status": "passed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "",
        }
    except (CellExecutionError, CellTimeoutError, Exception) as exc:
        text = str(exc)
        if len(text) > 2000:
            text = text[-2000:]
        return {
            "code": code,
            "file": str(path),
            "status": "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": text,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="content/notebooks")
    parser.add_argument("--report", default="reports/final_execution_report.json")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--engine", choices=["kernel", "inprocess"], default="kernel")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    root = Path(args.root).resolve()
    files = sorted(root.glob("*/*.ipynb"))
    if len(files) != 200:
        raise SystemExit(f"Expected 200 canonical notebooks; found {len(files)}.")

    runtime_dir = Path("/tmp/srai-notebook-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "data").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir / "matplotlib"))
    os.environ.setdefault("JUPYTER_DATA_DIR", str(runtime_dir / "jupyter"))
    os.environ.setdefault("XDG_CONFIG_HOME", str(runtime_dir / "config"))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(
                execute_inprocess if args.engine == "inprocess" else execute_kernel,
                path,
                *(() if args.engine == "inprocess" else (args.timeout,)),
            ): path
            for path in files
        }
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f"{result['status'].upper():6} {result['code']} {result['duration_seconds']:.3f}s")

    results.sort(key=lambda row: row["code"])
    passed = sum(row["status"] == "passed" for row in results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "engine": args.engine,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("total", "passed", "failed")}))


if __name__ == "__main__":
    main()
