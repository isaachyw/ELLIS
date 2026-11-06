"""Exact phase-guard references: a small Clifford prefix and a native fallback.

The native plan is compiled once. Eligible shots use a small prefix tableau,
a Pauli gauge, and CSS-tail propagation; other shots use the full masked
tableau without building or parsing another circuit. No loss-weight cutoff is
used. General Clifford/CSS splits and cultivation's terminal Pauli queries are
supported. The plans are ellis._core.Plan and ellis._core.FastPlan (src/ellis/reference).
"""
import numpy as np
import stim

from ellis import _core
from ellis import frame as mf
from ellis.msc.plan import compile_plan


class CultivationReference:
    """Exact plain-mode references for Clifford/CSS splits, including cultivation."""

    same_choice = False                 # the guarded suffix may choose another valid record

    def __init__(self, circuit: stim.Circuit, reload: bool = False):
        self.circuit, self.reload = circuit.flattened(), reload
        mf.validate(self.circuit)
        skeleton = mf.reference_skeleton(self.circuit)
        self.metadata = compile_plan(self.circuit)
        if self.metadata is None:
            self.plan = _core.Plan(skeleton, reload, self.circuit.num_qubits)
        else:
            self.plan = _core.FastPlan(skeleton, reload, self.circuit.num_qubits,
                                        self.metadata["cut"], self.metadata["boundary"],
                                        self.metadata["core_qubits"], self.metadata["queries"])
        self.stats = {"queries": 0, "fast": 0, "fallback": 0}

    def _record(self, result):
        record, fast = result if self.metadata is not None else (result, False)
        self.stats["queries"] += 1
        self.stats["fast" if fast else "fallback"] += 1
        return record

    def get(self, losses):
        """Return one valid noiseless reference for this exact loss pattern."""
        record = self._record(self.plan.reference(losses))
        return np.frombuffer(record, dtype=np.bool_)

    def get_many(self, patterns):
        """Compute a reference per shot with one native batch call."""
        records = [self._record(result) for result in self.plan.batch_reference(patterns)]
        return np.frombuffer(b"".join(records), dtype=np.bool_).reshape(len(records), self.circuit.num_measurements)
