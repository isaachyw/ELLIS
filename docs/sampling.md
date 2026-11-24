# Sampling API reference

`ellis.compile_sampler` turns a Stim circuit with atom loss into a `Sampler`, and `Sampler.sample` returns the
measurement record, detection events, observables, and heralds of each shot. This page covers
`ellis.compile_sampler`, `ellis.Sampler`, `ellis.SampleResult`, the two sampling routes, and the C++ engine
`ellis._core.MaskFrameSampler`. The sinter adapter `ellis.sinter` is under development.

```python
import ellis
from ellis.noise import surface_memory

circuit = surface_memory(d=5, rounds=5, p=1e-3, p_loss=1e-3)
sampler = ellis.compile_sampler(circuit, seed=1)
sampler.method       # "frame"
sampler.reason       # "one reference proven exact by frame-css"
r = sampler.sample(10**6, bit_packed=True, threads=8)
r.detectors.shape    # (1000000, ceil(num_detectors / 8))
```

## Circuits

The input is a `stim.Circuit`. `compile_sampler` flattens it, so `REPEAT` blocks are allowed. Two tagged `I_ERROR`
instructions add loss and reload. Stim parses both and treats them as no-ops.

| Instruction | Meaning |
| --- | --- |
| `I_ERROR[loss](p) q...` | Each atom `q` is lost with probability `p`. |
| `I_ERROR[reload](p) q...` | Each empty slot `q` gets a fresh atom in \|0> with probability `p` (default 1). A present atom is left alone. |

Both routes give a lost atom the same semantics (gate drop):

- A single-qubit gate or noise channel on a lost atom has no effect.
- A two-qubit gate, `DEPOLARIZE2`, or `PAULI_CHANNEL_2` with a lost operand is dropped. The present partner is unchanged.
- A measurement of a lost atom returns a uniformly random bit, and its herald is 1.
- `R`, `RX`, `RY`, and the reset part of `MR`, `MRX`, and `MRY` do not restore a lost atom. Only `I_ERROR[reload]` does.

Supported operations:

- Every unitary one- and two-qubit Clifford gate Stim knows.
- `X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`, and `I_ERROR`. An `I_ERROR` without the `loss` or `reload` tag does nothing.
- `R`, `RX`, `RY`, `M`, `MX`, `MY`, `MR`, `MRX`, and `MRY`, with an optional flip probability and inverted targets (`M !q`).
- `MPAD`, `DETECTOR`, `OBSERVABLE_INCLUDE`, `QUBIT_COORDS`, `SHIFT_COORDS`, and `TICK`.

`compile_sampler` raises `ValueError` on anything else, including `MPP`, `HERALDED_ERASE`, classically controlled
gates (`CX rec[-1] q`), and sweep bits. The check is `ellis.frame.validate`.

## `ellis.compile_sampler`

```python
ellis.compile_sampler(circuit, *, method="auto", certificate=None, certify_options=None, seed=None) -> Sampler
```

| Parameter | Meaning |
| --- | --- |
| `circuit` | A `stim.Circuit` that passes `ellis.frame.validate`. |
| `method` | `"auto"`, `"frame"`, or `"clifft"`. Any other value raises `ValueError`. |
| `certificate` | A `Certificate` from `ellis.certify.certify`. With `method="auto"`, it replaces the certify step. |
| `certify_options` | Keyword arguments for `certify` when `method="auto"` and `certificate` is `None`. `model` defaults to `MARKED`. |
| `seed` | Seed of the sampler's random stream. `None` draws one from the OS. |

### Routing

| `method` | Certify | Route |
| --- | --- | --- |
| `"auto"` | `certificate`, or `certify(circuit, model=MARKED, **certify_options)` | The C++ masked frame with `certificate.reference` when the verdict is `"proven"` and no gauge Paulis are needed. Clifft online otherwise. |
| `"frame"` | None | The C++ masked frame. It uses `certificate.reference` when `certificate` is proven without gauge Paulis, and otherwise Stim's reference sample of the loss-free circuit, which is not verified to be exact. |
| `"clifft"` | None | Clifft online. |

`MARKED` places losses and reloads only at the circuit's own markers, which is where sampling puts them. A circuit
without markers is therefore proven with any reference that is exact without loss.

`certify` returns one of four verdicts: `"proven"`, `"contextual"` (no single reference is exact; the certificate
carries a witness), `"consistent_on_family"` (no proof, but no counterexample on the loss patterns tried), and
`"outside-assumptions"`. Only `"proven"` without gauge Paulis routes to the masked frame. A `"proven"` certificate
from the `"diagonal-blocks+gauge"` method needs gauge Paulis, which the C++ engine does not apply, so it routes to
Clifft.

`certify` can take long on circuits that need its exact search. To compile one circuit more than once, certify it
once and pass the certificate:

```python
from ellis.certify import MARKED, certify

cert = certify(circuit, model=MARKED)
a = ellis.compile_sampler(circuit, certificate=cert, seed=1)
b = ellis.compile_sampler(circuit, certificate=cert, seed=2)
```

### Errors

| Exception | When |
| --- | --- |
| `ValueError` | `circuit` uses an unsupported operation, or `method` is not one of the three values. |
| `ImportError` | The route is Clifft and the `clifft` package is not installed. The message names the routing reason and the extra to install. |

## `ellis.Sampler`

| Member | Meaning |
| --- | --- |
| `method` | `"frame"` or `"clifft"`. |
| `reason` | Why `compile_sampler` chose the route, for example `"one reference proven exact by frame-css"` or `"certify verdict 'contextual' (exact-search): no reference is proven exact"`. |
| `certificate` | The `Certificate` used for routing, or `None` when `method` was `"frame"` or `"clifft"`. |
| `num_qubits`, `num_measurements`, `num_detectors`, `num_observables` | Sizes of the flattened circuit. |
| `sample(shots, *, bit_packed=False, threads=1)` | Returns a `SampleResult` for `shots` shots. |

`sample` runs one simulation pass per call and fills all four fields of the result from it. `threads` spreads shots
across threads. Results for a given seed do not depend on `threads`. Each call continues the sampler's random stream,
so two calls return different shots.

## `ellis.SampleResult`

A frozen dataclass with four NumPy arrays. Row `s` of each array belongs to shot `s`.

| Field | Columns | Value |
| --- | --- | --- |
| `measurements` | one per measurement, in record order | The measured bit. A lost readout is a uniformly random bit. |
| `detectors` | one per `DETECTOR` | 1 when the detector's parity differs from its value in Stim's noiseless reference sample, as in Stim's detector sampler. |
| `observables` | one per observable index | 1 when the observable differs from its value in Stim's noiseless reference sample. |
| `heralds` | one per measurement, same as `measurements` | 1 when that measurement read a lost atom. |

Array layout:

| `bit_packed` | dtype | Shape |
| --- | --- | --- |
| `False` (default, as in Stim) | `bool` | `(shots, n)` |
| `True` | `uint8` | `(shots, ceil(n / 8))`, 8 bits per byte in little-endian bit order, as in Stim |

Facts about `heralds`:

- A herald marks a measurement, not a loss event. An atom that is lost and never measured again sets no herald. An atom measured three times while lost sets three.
- After a successful reload, later measurements of the atom have herald 0.
- `MPAD` columns always have herald 0.
- `measurements[heralds]` is the set of placeholder bits.

## Routes

### C++ masked frame

`ellis._core.MaskFrameSampler` is a Pauli-frame simulator built on Stim's `FrameSimulator`. It keeps one alive bit per
atom and shot, and every gate is masked by it. All shots share one reference record, the certified one when the route
comes from `"auto"`. It simulates 256 shots per AVX2 word, in batches of up to 2048 shots, and runs Stim's
`STORE_EVERYTHING_TO_MEMORY` mode, so detectors are computed during the simulation.

### Clifft online

The Clifft route uses `clifft.noncomp` 0.13.0 in `execution="online"` mode, which simulates each shot's own lossy
circuit one shot at a time, so it needs no reference. It is an optional dependency:

```bash
uv sync --extra clifft         # or: pip install 'ellis[clifft]'
```

`ellis._clifft` translates the circuit before handing it to Clifft:

- `I_ERROR[loss](p)` becomes `LOSS(p)`.
- `I_ERROR[reload](p)` becomes a named `LEVEL_TRANSITION` that moves the `lost` level to `g` with probability `p`. All other entries of the transition matrix are 0. In Clifft a diagonal 1 is a jump onto the same level, which would collapse a present atom.
- Tags are removed from other instructions, because Clifft accepts tags only on `LEVEL_TRANSITION`.
- The model uses the classifier `[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,1,1]]`, whose third row heralds a readout of a leaked or lost level, and `reset_restores_lost=False`.

Clifft reports raw detector and observable parities. `ellis._clifft` XORs them with constants computed once from
Stim, so both routes report the same convention.

## Performance

Measured on one AMD EPYC 7V13 core with `surface_memory(d, rounds=d, p=1e-3, p_loss=1e-3)`, best of three runs:

| Route | Circuit | Shots | `bit_packed=True` | `bit_packed=False` |
| --- | --- | --- | --- | --- |
| frame | d=7 (385 measurements) | 200,000 | 0.049 s | 0.090 s |
| frame | d=15 (3585 measurements) | 50,000 | 0.120 s | 0.198 s |
| clifft | d=5 | 12,000 | 0.638 s | 0.633 s |
| clifft | d=7 | 4,000 | 0.756 s | 0.764 s |

On the frame route, packed output is faster because the simulation is the same and the output is 8 times smaller.
Unpacked output spends the difference expanding bits into bytes and writing them. On the Clifft route, the per-shot
simulation dominates and packing makes no difference. At d=7, Clifft samples about 5 kshots/s and the masked frame
about 4000 kshots/s.

## `ellis._core.MaskFrameSampler`

The C++ engine without certify or routing. The tests and `tools/bench/bench_frame.py` use it directly.

```python
ellis._core.MaskFrameSampler(circuit, *, reference=None, seed=None, batch_size=0, simd_width=0)
```

| Member | Meaning |
| --- | --- |
| `reference` (argument) | The shared reference record, `num_measurements` bits. `None` uses Stim's reference sample of the circuit without loss markers and without Pauli gates. |
| `batch_size` (argument) | Shots per batch, a power of two at least the SIMD width. 0 picks the largest batch, up to 2048, whose frame fits in 512 KB. |
| `simd_width` (argument) | 256, 128, or 64. 0 picks the widest width compiled in. |
| `sample(shots, *, bit_packed=False, threads=1)` | Returns `(measurements, detectors, observables, heralds)`. |
| `batch_size` (property) | The batch size in use. |
| `reference`, `num_qubits`, `num_measurements`, `num_detectors`, `num_observables`, `num_instructions`, `simd_width` | Read-only properties. |

Results depend on `seed`, `batch_size`, and `simd_width`, and not on `threads`.

## Removed from main

The NumPy reference sampler `ellis.frame.sample` and its research options (`gauge`, `canonical`, `ref_table`,
`reference_blocks`, `reference_provider`, `return_losses`, `software_feedforward`, `lost_partner_idle_p`, and
`forced_losses`) are on branch `frame-css-worst-case`. To force a loss in every shot, insert `I_ERROR[loss](1)` markers.
`ellis.check_exact.mark_losses(circuit, losses)` does this.
