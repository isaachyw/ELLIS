"""Masked Pauli-frame semantics of Clifford circuits with atom loss (gate-drop), shared by the samplers and ellis.certify.

A shot with loss runs its own Clifford circuit C_s = C_0 minus every operation on a qubit after it is lost.
The C++ engine (ellis._core.MaskFrameSampler) simulates it with a Stim frame whose gates are AND-masked by per-shot
"alive" bits, plus one reference record of C_0, which is exact whenever that record is also a valid noiseless
sample of every C_s (ellis.certify decides this).  A lost qubit is frozen: nothing touches its frame again, and its
readouts are heralded and filled with a uniformly random placeholder bit (Clifft's convention).

Loss instruction: `I_ERROR[loss](p) q...` (ppvm spelling; Stim parses and ignores it).
Reload instruction: `I_ERROR[reload](p) q...` loads a fresh atom in |0> into each empty slot q with
probability p (default 1) and leaves a present atom alone; a failed reload leaves the slot empty, which is
the same as a successful reload followed by a loss right after the marker (how explicit loss patterns write
it).  It can stand anywhere; to reload at a reset, write the marker just before the reset (before R / RX /
RY, or between the readout and reset of a split MR / MRX / MRY), so the fresh atom is reset like any other.
"""
import stim

ANNOTATIONS = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}
MEAS = {"M": "Z", "MZ": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRZ": "Z", "MRX": "X", "MRY": "Y"}
RESET = {"R": "Z", "RZ": "Z", "RX": "X", "RY": "Y"}


PAULI = {"X", "Y", "Z"}
UNITARY_1Q = {n for n, g in stim.gate_data().items() if g.is_unitary and g.is_single_qubit_gate}
UNITARY_2Q = {n for n, g in stim.gate_data().items() if g.is_unitary and g.is_two_qubit_gate}
PAULI_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "I_ERROR"}
SUPPORTED = ANNOTATIONS | set(MEAS) | set(RESET) | UNITARY_1Q | UNITARY_2Q | PAULI_NOISE


def is_loss(inst) -> bool:
    return inst.name == "I_ERROR" and inst.tag == "loss"


def is_reload(inst) -> bool:
    return inst.name == "I_ERROR" and inst.tag == "reload"


def validate(circuit: stim.Circuit) -> None:
    """Raise ValueError if the circuit uses an operation the masked frame cannot simulate exactly."""
    for k, inst in enumerate(circuit.flattened()):
        if inst.name in ANNOTATIONS:
            continue
        if is_reload(inst) and len(inst.gate_args_copy()) > 1:
            raise ValueError(f"instruction {k} `{inst}`: `I_ERROR[reload]` takes at most one probability")
        if inst.name not in SUPPORTED:
            raise ValueError(f"instruction {k} `{inst}`: `{inst.name}` is not supported by the masked frame")
        for t in inst.targets_copy():
            if t.is_measurement_record_target or t.is_sweep_bit_target:
                raise ValueError(f"instruction {k} `{inst}`: classically controlled operations are not supported")


def strip_loss(circuit: stim.Circuit) -> stim.Circuit:
    """Reference circuit: loss and reload markers removed and Pauli gates removed (they live in the frame
    instead, so that a Pauli on a lost qubit is masked like any other gate)."""
    return stim.Circuit("\n".join(str(inst) for inst in circuit.flattened()
                                  if not (is_loss(inst) or is_reload(inst)) and inst.name not in PAULI))


def explicit_lossy_circuit(circuit: stim.Circuit, losses) -> stim.Circuit:
    """C_s for one loss pattern, as an ordinary Stim circuit.
    losses = {instruction_index: [qubits lost right after that instruction]}; index -1: lost from the start.
    Operations on lost qubits are removed; a lost readout becomes a fresh |+> ancilla measured
    in Z (uniform placeholder bit), so the measurement record keeps its layout.  A reload marker on an
    empty slot becomes `R q`."""
    # Built as text and parsed once: Stim's Circuit.append(name, targets, args) is slow where C++ unwinding is (macOS).
    anc = circuit.num_qubits
    out: list[str] = []
    lost = set(losses.get(-1, ()))
    for k, inst in enumerate(circuit.flattened()):
        name, targets = inst.name, inst.targets_copy()
        qs = [t.value for t in targets]
        args = inst.gate_args_copy()
        head = name + (f"({','.join(map(repr, args))})" if args else "")
        if is_reload(inst):
            for q in qs:
                if q in lost:
                    lost.discard(q)
                    out.append(f"R {q}")
        elif name in ANNOTATIONS or is_loss(inst):
            if name != "I_ERROR":
                out.append(str(inst))
        elif name in MEAS:
            for t, text in zip(targets, str(inst).split(" ")[1:]):
                if t.value in lost:
                    out.append(f"RX {anc}\nM {anc}")
                else:
                    out.append(f"{head} {text}")     # keeps an inverted target `!q`
        else:
            two = stim.gate_data(name).is_two_qubit_gate
            groups = list(zip(qs[::2], qs[1::2])) if two else [(q,) for q in qs]
            keep = [q for g in groups if not (set(g) & lost) for q in g]
            if keep:
                out.append(f"{head} {' '.join(map(str, keep))}")
        lost |= set(losses.get(k, ()))
    return stim.Circuit("\n".join(out))
