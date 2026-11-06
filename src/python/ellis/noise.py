"""Circuit-level noise and atom loss.

noisy: DEPOLARIZE1(p) after 1q gates, DEPOLARIZE2(p) + loss(p_loss) on each operand after 2q gates,
reset/measurement flips p; instructions tagged `noiseless` get no noise and no loss. No reload.
surface_memory: Stim's rotated surface-code memory with loss after every CX's DEPOLARIZE2.
"""
import stim

P, P_LOSS = 1e-3, 2e-3
ONE_Q = {"H", "S", "S_DAG"}
TWO_Q = {"CX", "CZ", "SWAP"}
RESET_ERR = {"R": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}
MEAS = {"M", "MX", "MY", "MR", "MRX", "MRY"}


LOSS = "I_ERROR[loss]({p}) {qs}"


def noisy(circ: stim.Circuit, loss_spelling: str = LOSS, p_loss: float = P_LOSS, p2: float | None = None, p: float = P) -> str:
    """p: strength of every Pauli channel; p2: DEPOLARIZE2 strength after 2q gates (default p).
    loss_spelling: format string of the loss line after each 2q gate, with fields {p} and {qs}."""
    p2 = p if p2 is None else p2
    lines = []
    for inst in circ.flattened():
        name, tag = inst.name, inst.tag
        qs = " ".join(str(t.value) for t in inst.targets_copy())
        quiet = tag == "noiseless"
        if name in MEAS and not quiet:
            lines.append(f"{name}({p}) {qs}")
            continue
        lines.append(str(inst))
        if quiet:
            continue
        if name in ONE_Q:
            lines.append(f"DEPOLARIZE1({p}) {qs}")
        elif name in TWO_Q:
            lines.append(f"DEPOLARIZE2({p2}) {qs}")
            lines.append(loss_spelling.format(p=p_loss, qs=qs))
        elif name in RESET_ERR:
            lines.append(f"{RESET_ERR[name]}({p}) {qs}")
    return "\n".join(lines)


def surface_memory(d: int, rounds: int, p: float, p_loss: float) -> stim.Circuit:
    """Rotated surface-code memory-Z with uniform circuit-level depolarizing noise p; each CX operand is lost
    with probability p_loss right after the CX's DEPOLARIZE2."""
    base = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p).flattened()
    out = stim.Circuit()
    for inst in base:
        out.append(inst)
        if inst.name == "DEPOLARIZE2":
            out.append("I_ERROR", inst.targets_copy(), [p_loss], tag="loss")
    return out
