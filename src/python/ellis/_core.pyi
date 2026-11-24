"""Type stub for ellis._core, built from src/ellis and src/python/bindings.cc (see CMakeLists.txt)."""
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
        seed: int | None = None,
        batch_size: int = 0,
        simd_width: int = 0,
    ) -> None: ...
    def sample(
        self, shots: int, *, bit_packed: bool = False, threads: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(measurements, detectors, observables, heralds) from one simulation pass."""
    @property
    def batch_size(self) -> int: ...
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
    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(measurements, detectors, observables) from one simulation pass."""
    @property
    def simd_width(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
