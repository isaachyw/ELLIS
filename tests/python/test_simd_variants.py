"""ellis._core loads the fastest SIMD variant this CPU runs, and ELLIS_SIMD forces one (src/python/ellis/_core.py)."""
import importlib.util
import os
import subprocess
import sys

import pytest
from ellis._cpu import best_variant

from ellis import _core

ORDER = ["avx2", "sse2", "polyfill"]
WIDTHS = {"avx2": [256, 128, 64], "sse2": [128, 64], "polyfill": [64]}


def load(simd: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ELLIS_SIMD"} | ({"ELLIS_SIMD": simd} if simd else {})
    code = "import ellis._core as c; print(c.VARIANT, *c.SIMD_WIDTHS)"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_loads_a_variant_this_cpu_runs():
    assert ORDER.index(_core.VARIANT) >= ORDER.index(best_variant())
    assert _core.SIMD_WIDTHS == WIDTHS[_core.VARIANT]


@pytest.mark.parametrize("simd", ORDER)
def test_forcing_loads_a_variant_iff_it_is_installed_and_this_cpu_runs_it(simd):
    r = load(simd)
    usable = importlib.util.find_spec(f"ellis._core_{simd}") is not None and ORDER.index(simd) >= ORDER.index(best_variant())
    if usable:
        assert r.stdout.split() == [simd, *map(str, WIDTHS[simd])], r.stderr
    else:
        assert r.returncode != 0 and f"ELLIS_SIMD={simd!r}" in r.stderr


def test_forcing_an_unknown_variant_fails():
    r = load("avx512")
    assert r.returncode != 0 and "ELLIS_SIMD='avx512'" in r.stderr
