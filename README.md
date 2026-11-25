<p align="center">
  <img src="https://raw.githubusercontent.com/isaachyw/ELLIS/main/docs/assets/logos/ellis-logo-light.png" width="220" alt="ELLIS logo">
</p>

# ELLIS: Extremely Fast Quantum Circuit Simulation under Qubit Loss and Leakage

[![arXiv](https://img.shields.io/badge/Paper-2611.26173-B31B1B?style=flat-square)](https://arxiv.org/abs/2611.26173)

ELLIS (Efficient Loss and Leakage Invariant-reference Simulation) samples Clifford circuits with qubit loss and leakage
exactly. It uses a masked Pauli frame and one shared reference record, so each shot does not need its own
tableau reference.

- **One sampling API.** `ellis.compile_sampler` certifies the circuit. When one reference is proven exact, it samples
  with the C++ masked frame `ellis._core.MaskFrameSampler`, built on Stim's `FrameSimulator`. Otherwise it samples
  with Clifft's noncomputational sampler in online mode. Both drop a lost qubit's later gates and read its
  measurements as heralded fair coins.
- **Common-reference certificates.** `ellis.certify` proves that one reference is exact for every loss
  pattern by frame-CSS relabelling or closed S/CZ blocks, or returns a signed-parity contradiction that
  shows no such reference exists. Pass the circuit with its noise injected: losses and reloads happen at its
  `I_ERROR[loss]` and `I_ERROR[reload]` markers. A circuit with neither marker is certified under
  `PESSIMISTIC` (loss after every operation, |0> reload anywhere), so its reference holds under every loss
  model. An explicit `LossModel` overrides both. `frame_css` decides frame-CSS exactly; a circuit outside the
  assumptions that make the decision exact (classical feedback, multi-qubit measurements, a prefix with
  too many open frame choices) gets verdict `"outside-assumptions"` instead.

## Loss instructions

ELLIS reads Stim circuits and adds two tagged `I_ERROR` instructions. Stim parses both and ignores them.

| Instruction                 | Meaning                                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `I_ERROR[loss](p) q...`   | Each atom `q` is lost with probability `p`. Its later gates are dropped and its readouts become fair coins.   |
| `I_ERROR[reload](p) q...` | Each empty slot `q` gets a fresh atom in \|0> with probability `p` (default 1). A present atom is left alone. |

A reload happens only where its marker stands. To reload at a reset, write the marker just before it:

```
CX 0 1
I_ERROR[loss](0.001) 0 1
M 1
I_ERROR[reload] 1
R 1
```

## Sampling performance

<p>
  <img src="https://raw.githubusercontent.com/isaachyw/ELLIS/main/docs/assets/performance/sampling_single_core.png" width="49%" alt="Single-core sampling throughput for PPVM, Clifft online, and ELLIS on BB144 memory and the distance-7 transversal CNOT with atom loss.">
  <img src="https://raw.githubusercontent.com/isaachyw/ELLIS/main/docs/assets/performance/sampling_multicore_128.png" width="49%" alt="128-thread sampling throughput for PPVM, Clifft online, and ELLIS on BB144 memory and the distance-7 transversal CNOT with atom loss.">
</p>

## Install

```bash
uv add ellis-qec             # or: pip install ellis-qec
uv add "ellis-qec[sinter]"   # also sinter, for ellis.sinter
```

Import it as `ellis`. Wheels support CPython 3.12, 3.13, and 3.14 on Linux x86-64 and macOS arm64 (macOS 15 or
later).

### Build from source

```bash
uv sync                  # builds ellis._core with CMake (fetches Stim 1.16.0 once)
uv sync --extra sinter   # also sinter, for ellis.sinter
```

The build needs a C++20 compiler: GCC 10 or newer, or Apple clang.

## Use

```python
import ellis
from ellis.noise import surface_memory

circuit = surface_memory(d=5, rounds=5, p=1e-3, p_loss=1e-3)  # I_ERROR[loss](p) marks loss sites
sampler = ellis.compile_sampler(circuit, seed=1)  # certifies: verdict "proven", method "frame-css"
sampler.method  # "frame"
r = sampler.sample(10**4)  # r.measurements, r.detectors, r.observables, r.heralds
r = sampler.sample(10**6, bit_packed=True, threads=8)  # the same fields, bit-packed, on 8 threads
```

| `compile_sampler` route                                  | Sampler                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `method="auto"`, certificate proven without gauge Paulis | C++ masked frame with the certified reference                                        |
| `method="auto"`, any other certificate                   | Clifft online                                                                        |
| `method="frame"`                                         | C++ masked frame; Stim's reference (unverified) unless a proven certificate is given |
| `method="clifft"`                                        | Clifft online                                                                        |

`method="auto"` certifies with `ellis.certify.certify(circuit, model=MARKED, **certify_options)` unless
`certificate=` is given: sampling loses atoms only at the markers.
`sample` returns a `SampleResult` with all four fields from one simulation pass; `bit_packed=True` uses Stim's
layout. Detection events are relative to the noiseless reference sample, as in Stim.
[docs/sampling.md](https://github.com/isaachyw/ELLIS/blob/main/docs/sampling.md) is the full reference: supported operations, loss semantics, routing, result
layout, heralds, and performance.

The sinter adapter `ellis.sinter` is under development.

## Layout

| Path                     | Contents                                                                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/ellis/frame/`       | C++ masked frame: program compilation, `stim::FrameSimulator` with alive masks, batched sampler                                                                               |
| `src/python/`            | `bindings.cc` (the `ellis._core_<variant>` modules), `cpu.cc` (`ellis._cpu`, which picks the variant)                                                                         |
| `src/python/ellis/`      | `sampler` (`compile_sampler`), `_clifft` (Clifft route), `frame` (loss semantics), `certify/` (`frame_css`, `blocks`, `constraints`), `noise`, `sinter`, `gf2`, `check_exact` |
| `tests/python/`          | tests against Stim on explicit lossy circuits and between the frame and Clifft routes; fixtures in `tests/fixtures/`                                                          |
| `tools/`                 | `bench/bench_frame.py` (masked frame and Stim speed), `classify_lightstim.py` (certify each LightStim protocol), fixture exporters                                            |
| `cmake/`                 | Stim fetch, compiler selection, SIMD variants                                                                                                                                 |
| `.github/workflows/`     | `test.yml` (the tests on both x86-64 variants), `wheels.yml` (sdist and wheels; publishes to TestPyPI and PyPI on a `v*` tag, see `RELEASING.md`)                             |

## Test

```bash
uv run pytest -n 8
uv run python -m ellis.check_exact 'tests/fixtures/lightstim/*.stim'   # sampler vs Stim on random loss patterns
uv run python tools/classify_lightstim.py --benchmark                  # paper App. C classification
```

## License

Apache-2.0.

## Citation

If you find ELLIS useful in your work, please cite our paper:

```bibtex
@misc{hou2026ellis,
      title={Ellis: Extremely Fast Quantum Circuit Simulation under Qubit Loss and Leakage},
      author={Yuewen Hou and Gokul Subramanian Ravi},
      year={2026},
      eprint={2611.26173},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2611.26173},
}
```

For circuits without a proven common reference, such as circuits that are not frame-CSS, ELLIS samples with
[Clifft](https://github.com/unitaryfoundation/clifft)'s non-computational sampler in online mode, adapted to ELLIS's
loss and reload markers (`ellis._clifft`). If your results use that route (`sampler.method == "clifft"`), please also
cite Clifft:

```bibtex
@misc{chase2026clifftfastexactsimulation,
      title={Clifft: Fast Exact Simulation of Near-Clifford Quantum Circuits},
      author={Bradley A. Chase and Farrokh Labib},
      year={2026},
      eprint={2604.27058},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.27058},
}
```
