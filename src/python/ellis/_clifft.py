"""Clifft's noncomputational sampler in online mode, on ELLIS circuits: the sampling route for circuits without a
proven common reference.  Clifft simulates each shot's own lossy circuit, so it needs no reference.

Its semantics match the masked frame's: a two-qubit gate or two-qubit noise channel with a lost operand is dropped
and leaves the present partner alone (Clifft's default without gate_partner_effects), a lost readout is a heralded
uniform bit (the classifier below), resets do not restore a lost atom (reset_restores_lost=False), and only reload
markers do.  Clifft reports raw detector parities; ELLIS reports them relative to Stim's noiseless reference sample.
"""
import numpy as np
import stim
from clifft import noncomp

from ellis import frame as mf

G, LOST = 0, 4  # Clifft's levels: g, e, leak_g, leak_e, lost
CLASSIFIER = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1]]  # P[record 0, record 1, herald][level]


def reload_transition(p: float) -> list[list[float]]:
    """T[to][from]: an empty slot gets a fresh atom in |0> (level g) with probability p.  Every other column is zero,
    i.e. no jump: a diagonal 1 would be a jump onto the same level, which collapses a present atom's state."""
    t = [[0.0] * 5 for _ in range(5)]
    t[G][LOST] = p
    return t


def translate(circuit: stim.Circuit) -> tuple[str, dict[str, list[list[float]]]]:
    """Clifft text of the flattened circuit and its named reload transitions.  `I_ERROR[loss](p)` becomes `LOSS(p)`,
    `I_ERROR[reload](p)` a `LEVEL_TRANSITION[reload<k>]`, other `I_ERROR`s are dropped, and every other instruction
    is Stim text without its tag (Clifft accepts tags only on LEVEL_TRANSITION)."""
    lines, names = [], {}
    for inst in circuit.flattened():
        targets, args = inst.targets_copy(), inst.gate_args_copy()
        qs = " ".join(str(t.value) for t in targets)
        if mf.is_loss(inst):
            lines.append(f"LOSS({args[0]}) {qs}")
        elif mf.is_reload(inst):
            p = args[0] if args else 1.0
            lines.append(f"LEVEL_TRANSITION[{names.setdefault(p, f'reload{len(names)}')}] {qs}")
        elif inst.name != "I_ERROR":
            lines.append(str(stim.CircuitInstruction(inst.name, targets, args)))
    return "\n".join(lines), {name: reload_transition(p) for p, name in names.items()}


def _bits(a: np.ndarray, bit_packed: bool) -> np.ndarray:
    return np.packbits(a, axis=1, bitorder="little") if bit_packed else a.astype(bool)


class Backend:
    def __init__(self, circuit: stim.Circuit, seed: int | None):
        text, transitions = translate(circuit)
        model = noncomp.Model(classifier=noncomp.Classifier(CLASSIFIER), transitions=transitions, reset_restores_lost=False)
        self.program = noncomp.compile(text, model, execution="online")  # fixed execution rejects named transitions
        self._seeds = np.random.default_rng(seed)
        zero = np.zeros((1, circuit.num_measurements), bool)
        dets, obs = circuit.compile_m2d_converter().convert(measurements=zero, separate_observables=True)
        self._detector_flip, self._observable_flip = dets[0].astype(np.uint8), obs[0].astype(np.uint8)

    def sample(self, shots: int, bit_packed: bool, threads: int):
        r = self.program.sample(shots, seed=int(self._seeds.integers(2**63)), threads=threads)
        return (_bits(r.measurements, bit_packed), _bits(r.detectors ^ self._detector_flip, bit_packed),
                _bits(r.observables ^ self._observable_flip, bit_packed), _bits(r.heralds, bit_packed))
