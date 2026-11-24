"""ellis._core: the C++ masked frame (src/ellis, src/python/bindings.cc) from the fastest SIMD variant this CPU runs.

Each variant ellis._core_<name> is the same module compiled with its own machine flags and its own copy of Stim, whose
SIMD word width is fixed at compile time.  x86-64 wheels ship avx2 and sse2, other wheels and source builds one variant
(cmake/SimdVariants.cmake).  Set ELLIS_SIMD=avx2|sse2|polyfill to force a variant, e.g. to test it.
"""
import importlib
import importlib.util
import os

from ellis._cpu import best_variant

ORDER = ("avx2", "sse2", "polyfill")  # fastest first; a CPU runs its best variant and every later one


def _select() -> str:
    runnable = ORDER[ORDER.index(best_variant()):]
    built = [v for v in ORDER if importlib.util.find_spec(f"ellis._core_{v}") is not None]
    forced = os.environ.get("ELLIS_SIMD")
    if forced:
        if forced not in runnable or forced not in built:
            raise ImportError(f"ELLIS_SIMD={forced!r}: this CPU runs {list(runnable)} and this install has {built}")
        return forced
    for v in runnable:
        if v in built:
            return v
    raise ImportError(f"no ellis._core variant this CPU runs ({list(runnable)}) is installed (has {built}); "
                      "build ELLIS from source on this machine")


VARIANT = _select()
_module = importlib.import_module(f"ellis._core_{VARIANT}")
MaskFrameSampler = _module.MaskFrameSampler
StimSampler = _module.StimSampler
SIMD_WIDTHS: list[int] = _module.SIMD_WIDTHS
