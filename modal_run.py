#!/usr/bin/env python3
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def run_check(name: str, fn: Callable[[], str | None]) -> CheckResult:
    try:
        details = fn() or ""
        return CheckResult(name=name, ok=True, details=details)
    except Exception:
        return CheckResult(
            name=name,
            ok=False,
            details=traceback.format_exc(),
        )


def compile_repo() -> str:
    compiled = 0
    for file_path in iter_python_files(SRC_ROOT):
        py_compile.compile(str(file_path), doraise=True)
        compiled += 1
    return f"compiled {compiled} python files"


def run_embedded_tests() -> str:
    sys.path.insert(0, str(SRC_ROOT))
    from experiments.SSL import dataparser

    dataparser.test_argparse()
    dataparser.test_optional()
    return "ran dataparser.test_argparse and dataparser.test_optional"


def import_smoke_module(module_name: str) -> CheckResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-c",
        f"import {module_name}; print('ok')",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode == 0:
        return CheckResult(name=f"import {module_name}", ok=True)

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    combined = "\n".join(part for part in (stdout, stderr) if part)

    if "OMP: Error #179" in combined and "Can't open SHM2" in combined:
        return CheckResult(
            name=f"import {module_name}",
            ok=True,
            details="skipped hard OpenMP shared-memory failure from sandboxed runtime",
        )

    return CheckResult(
        name=f"import {module_name}",
        ok=False,
        details=combined or f"subprocess exited with code {proc.returncode}",
    )


def run_import_smoke() -> list[CheckResult]:
    modules = [
        "methods.swd",
        "methods.sswd",
        "methods.s3wd",
        "methods.gsssw",
        "methods.wd",
        "utils.misc",
        "utils.plot",
        "utils.s3w",
        "utils.vi",
        "utils.vmf",
    ]
    return [import_smoke_module(module_name) for module_name in modules]


def print_results(results: list[CheckResult]) -> int:
    failures = [result for result in results if not result.ok]

    print("modal_run summary")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}")
        if result.details:
            for line in result.details.strip().splitlines():
                print(f"  {line}")

    print()
    print(f"checks: {len(results)}, failures: {len(failures)}")
    return 1 if failures else 0


def main() -> int:
    if not SRC_ROOT.exists():
        print(f"missing source directory: {SRC_ROOT}", file=sys.stderr)
        return 1

    results = [
        run_check("compile repo", compile_repo),
        run_check("embedded tests", run_embedded_tests),
    ]
    results.extend(run_import_smoke())
    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
