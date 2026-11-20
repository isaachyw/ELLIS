"""Type stub for ellis._core, built from src/ellis and src/python/bindings.cc (see CMakeLists.txt)."""
from typing import Any

import numpy as np
import numpy.typing as npt
import stim

SIMD_WIDTHS: list[int]

class MaskFrameSampler:
    """Masked Pauli-frame sampler for a circuit with `I_ERROR[loss](p)` loss (gate-drop semantics)."""

    def __init__(
        self,
        circuit: stim.Circuit | str,
        *,
        reference: npt.ArrayLike | None = None,
        forced_losses: dict[int, list[int]] | None = None,
        seed: int | None = None,
        batch_size: int = 0,
        simd_width: int = 0,
    ) -> None: ...
    def sample(self, shots: int, *, bit_packed: bool = False, heralds: bool = True, threads: int = 1) -> Any:
        """(measurements, heralds), or just measurements with heralds=False."""
    def sample_detectors(
        self, shots: int, *, bit_packed: bool = False, heralds: bool = False, threads: int = 1
    ) -> tuple[np.ndarray, ...]:
        """(detectors, observables), plus per-measurement heralds with heralds=True."""
    def batch_size(self, *, detectors: bool = False, heralds: bool = True) -> int: ...
    @property
    def reference(self) -> np.ndarray: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_measurements(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    @property
    def num_instructions(self) -> int: ...
    @property
    def simd_width(self) -> int: ...

class StimSampler:
    """stim::FrameSimulator (no loss) in the same batching and output code: the speed baseline."""

    def __init__(
        self, circuit: stim.Circuit | str, *, seed: int | None = None, batch_size: int = 0, simd_width: int = 0
    ) -> None: ...
    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> np.ndarray: ...
    def sample_detectors(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def simd_width(self) -> int: ...
    @property
    def batch_size(self) -> int: ...

type LossPattern = dict[int, list[int]]
type QueryInput = tuple[int, list[int], str, int, bool]

class Plan:
    """Exact noiseless reference of each loss pattern from the full tableau of its surviving circuit."""

    def __init__(self, skeleton: Any, num_qubits: int) -> None: ...
    def reference(self, losses: LossPattern) -> bytes: ...
    def batch_reference(self, patterns: list[LossPattern]) -> list[bytes]: ...

class FastPlan:
    """MSC S-proxy reference: prefix tableau, phase check, CSS gauge sweep and terminal queries, with the full
    tableau as the exact fallback.  reference() returns (record, fast path taken)."""

    def __init__(
        self,
        skeleton: Any,
        num_qubits: int,
        cut: int,
        boundary: int,
        core_qubits: list[int],
        queries: list[QueryInput],
    ) -> None: ...
    def reference(self, losses: LossPattern) -> tuple[bytes, bool]: ...
    def batch_reference(self, patterns: list[LossPattern]) -> list[tuple[bytes, bool]]: ...
