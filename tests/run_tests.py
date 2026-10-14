#!/usr/bin/env python3
"""Minimal test runner for environments without pytest.

Discovers every ``test_*`` function in ``tests/test_*.py``, runs it, and
reports pass/fail. Exit code 1 if any test fails (useful in CI).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    failures = 0
    total = 0
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            total += 1
            t0 = time.perf_counter()
            try:
                getattr(module, name)()
                print(f"PASS  {path.stem}::{name}  ({time.perf_counter() - t0:.2f} s)")
            except Exception:                                   # noqa: BLE001
                failures += 1
                print(f"FAIL  {path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
