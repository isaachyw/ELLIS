"""One sampling API for Clifford circuits with atom loss.

    s = ellis.compile_sampler(circuit)   # certifies the circuit, then picks the sampler
    r = s.sample(10**5)                  # SampleResult: measurements, detectors, observables, heralds

When ellis.certify proves that one reference record is exact for every loss pattern, the circuit samples with the C++
masked frame (ellis._core.MaskFrameSampler) and that reference.  Otherwise it samples with Clifft's noncomputational
sampler in online mode (ellis._clifft, the optional `clifft` extra), which simulates each shot's own lossy circuit.
Both use gate-drop loss semantics; see ellis.frame.
"""
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import stim

from ellis import _clifft
from ellis import frame as mf
from ellis._core import MaskFrameSampler
from ellis.certify import MARKED, Certificate, certify

Method = Literal["frame", "clifft"]


@dataclass(frozen=True)
class SampleResult:
    """Per-shot rows: bool (shots, n) arrays, or with bit_packed uint8 (shots, ceil(n / 8)) arrays in little-endian bit
    order (Stim's layout).  Detectors and observables are relative to Stim's noiseless reference sample, as in Stim;
    heralds[s, i] is 1 when measurement i of shot s read a lost qubit."""
    measurements: np.ndarray
    detectors: np.ndarray
    observables: np.ndarray
    heralds: np.ndarray


def route(certificate: Certificate) -> tuple[Method, str]:
    """The sampler a certificate allows, and why."""
    if certificate.verdict != "proven" or certificate.reference is None:
        return "clifft", f"certify verdict {certificate.verdict!r} ({certificate.method}): no reference is proven exact"
    if certificate.gauge is not None:
        return "clifft", f"proven by {certificate.method}, whose gauge Paulis the C++ masked frame does not apply"
    return "frame", f"one reference proven exact by {certificate.method}"


class _Backend(Protocol):
    def sample(self, shots: int, bit_packed: bool, threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(measurements, detectors, observables, heralds)."""


class _Frame:
    def __init__(self, circuit: stim.Circuit, certificate: Certificate | None, seed: int | None):
        exact = certificate is not None and route(certificate)[0] == "frame"
        self._sampler = MaskFrameSampler(circuit, reference=certificate.reference if exact else None, seed=seed)

    def sample(self, shots, bit_packed, threads):
        return self._sampler.sample(shots, bit_packed=bit_packed, threads=threads)


_BACKENDS = {"frame": _Frame, "clifft": lambda circuit, certificate, seed: _clifft.Backend(circuit, seed)}


class Sampler:
    """A compiled sampler; `method` names its backend and `reason` says why it was chosen."""

    def __init__(self, circuit: stim.Circuit, method: Method, reason: str, certificate: Certificate | None,
                 backend: _Backend):
        self.method, self.reason, self.certificate = method, reason, certificate
        self.num_qubits, self.num_measurements = circuit.num_qubits, circuit.num_measurements
        self.num_detectors, self.num_observables = circuit.num_detectors, circuit.num_observables
        self._backend = backend

    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> SampleResult:
        """Measurements, detectors, observables and heralds of `shots` shots, from one simulation pass."""
        return SampleResult(*self._backend.sample(shots, bit_packed, threads))


def compile_sampler(circuit: stim.Circuit, *, method: Literal["auto", "frame", "clifft"] = "auto",
                    certificate: Certificate | None = None, certify_options: dict | None = None,
                    seed: int | None = None) -> Sampler:
    """A sampler for a Clifford circuit with `I_ERROR[loss](p)` / `I_ERROR[reload](p)` markers (see ellis.frame).

    method="auto" routes by `certificate`, or by certify(circuit, model=MARKED, **certify_options) when none is given
    (sampling loses atoms only at the markers, so a circuit without markers is proven trivially): the C++ masked
    frame with the certified reference when it is proven exact without gauge Paulis, Clifft otherwise.
    method="frame" skips certify and uses a proven certificate's reference if given, else Stim's reference of the
    loss-free circuit, which is unverified (for speed comparisons).  method="clifft" always samples with Clifft.
    seed: seed of the sampler's random stream (None: from the OS); each sample call continues it.
    Raises ValueError for an operation outside Clifford gates, Pauli noise, resets and single-qubit measurements, and
    ImportError when the circuit needs Clifft and it is not installed."""
    if method not in ("auto", *_BACKENDS):
        raise ValueError(f"method must be 'auto', 'frame' or 'clifft', not {method!r}")
    circuit = circuit.flattened()
    mf.validate(circuit)
    if method == "auto":
        if certificate is None:
            certificate = certify(circuit, **{"model": MARKED, **(certify_options or {})})
        method, reason = route(certificate)
    else:
        reason = f"method={method!r} requested"
    try:
        backend = _BACKENDS[method](circuit, certificate, seed)
    except ModuleNotFoundError as e:
        if e.name != "clifft":
            raise
        raise ImportError(f"this circuit samples with Clifft ({reason}), which is not installed: "
                          "pip install 'ellis[clifft]'") from e
    return Sampler(circuit, method, reason, certificate, backend)
