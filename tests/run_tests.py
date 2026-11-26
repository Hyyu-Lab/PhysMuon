"""Minimal runner (no pytest needed): python tests/run_tests.py"""
import importlib
import pathlib
import sys
import traceback

here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]          # tests/ (for the reference) and the repository root
mod = importlib.import_module("test_physmuon")
names = sorted(n for n in dir(mod) if n.startswith("test_"))
ok = 0
for name in names:
    try:
        getattr(mod, name)()
        print(f"PASS {name}")
        ok += 1
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()
print(f"{ok}/{len(names)} passed")
sys.exit(0 if ok == len(names) else 1)
