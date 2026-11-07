# ELLIS

ELLIS (Efficient Loss and Leakage Invariant-reference Simulation) samples Clifford circuits with atom loss
exactly. It uses a masked Pauli frame and one shared reference record, so each shot does not need its own
tableau reference.

- **Masked frame sampler.** `ellis.frame` and the C++ engine `ellis._core.MaskFrameSampler`, built on
  Stim's `FrameSimulator`. A lost qubit drops its later gates, and its readouts become fair coins.
- **Common-reference certificates.** `ellis.certify` proves that one reference is exact for every loss
  pattern (frame-CSS relabelling, closed S/CZ blocks), or returns a signed-parity contradiction that
  shows no such reference exists.
- **MSC S-proxy references.** `ellis.msc.reference.CultivationReference` computes a reference for each loss pattern
  from a small prefix tableau and a CSS gauge sweep, and falls back to the full tableau when its checks
  fail.

## Install

```bash
uv sync                  # builds ellis._core with CMake (fetches Stim 1.16.0 once)
uv sync --extra sinter   # also sinter, for ellis.sinter
```

The build needs GCC 10 or newer and targets AVX2 by default. For a CPU without AVX2, run
`uv sync -C cmake.define.SIMD_WIDTH=128`. On Perlmutter, CMake selects the `gcc-native` module's
`g++` automatically. Run the first build on a compute node.

## Use

```python
import stim
from ellis import frame
from ellis.certify import certify
from ellis.noise import surface_memory

circuit = surface_memory(d=5, rounds=5, p=1e-3, p_loss=1e-3)   # I_ERROR[loss](p) marks loss sites
cert = certify(circuit)            # losses after 2q gates (LossModel()); verdict "proven", method "frame-css"
sampler = frame.compile_sampler(circuit, reference=cert.reference, seed=1)
dets, obs = sampler.sample_detectors(10**6, bit_packed=True, threads=8)
```

For a circuit without a common reference, such as an MSC S-proxy circuit, pass a per-shot provider to the
numpy sampler:

```python
from ellis.msc.reference import CultivationReference

provider = CultivationReference(circuit, reload=False)
meas, heralds = frame.sample(circuit, 1000, reference_provider=provider)
```

`ellis.sinter.masked(decoder)` is a `sinter.Sampler`. It samples with the masked frame and decodes with any
sinter decoder.

## Layout

| Path | Contents |
|---|---|
| `src/ellis/frame/` | C++ masked frame: program compilation, `stim::FrameSimulator` with alive masks, batched sampler |
| `src/ellis/reference/` | C++ exact references for a loss pattern: full tableau (`Plan`) and MSC prefix (`FastPlan`) |
| `src/python/bindings.cc` | the `ellis._core` module |
| `src/python/ellis/` | `frame`, `certify/` (`frame_css`, `blocks`, `constraints`), `msc/`, `noise`, `sinter`, `gf2`, `check_exact` |
| `tests/python/` | tests against Stim on explicit lossy circuits; fixtures in `tests/fixtures/` |
| `tools/` | `bench/bench_frame.py` (masked frame and Stim speed), `classify_lightstim.py` (certify each LightStim protocol), fixture exporters |
| `cmake/` | Stim fetch, compiler selection, SIMD flags |

## Test

```bash
uv run pytest
uv run python -m ellis.check_exact 'tests/fixtures/lightstim/*.stim'   # sampler vs Stim on random loss patterns
uv run python tools/classify_lightstim.py --benchmark                  # paper App. C classification
```
