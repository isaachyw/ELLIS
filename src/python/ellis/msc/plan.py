"""Plans for a Clifford prefix followed by a CSS tail and optional Pauli queries."""
import stim

from ellis import frame as mf

_ANNOTATIONS = {"QUBIT_COORDS", "TICK", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
_FRAME = {"X", "Y", "Z", "X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "I_ERROR"}
_PHYSICAL = {"CX", "S", "S_DAG", "R", "RX", "M", "MX", "MR", "MRX"}
_MEASURE = {"M", "MX", "MR", "MRX"}
_CSS = {"CX", "SWAP", "I", "R", "RX", "M", "MX", "MR", "MRX"}


def compile_plan(circuit):
    """Prefer the cultivation query plan, then try a general Clifford/CSS split.

    The general split treats every non-CSS operation as part of the prefix,
    regardless of tags. It needs no terminal-query rule or circuit rewriting.
    """
    plan = _cultivation_plan(circuit)
    if plan is not None:
        return plan
    skeleton = mf.reference_skeleton(circuit)
    phase = [k for k, kind, name, _ in skeleton if kind != "pad" and name not in _CSS]
    cut = phase[-1] + 1 if phase else 0
    # Checking a nearly complete tableau costs more than its final readouts.
    # Require CSS two-qubit work to bypass, unless the whole circuit is CSS.
    if phase and not any(k >= cut and kind == "two" for k, kind, _, _ in skeleton):
        return None
    parents = list(range(circuit.num_qubits))

    def root(q):
        while parents[q] != q:
            parents[q] = parents[parents[q]]
            q = parents[q]
        return q

    seeds = set()
    for k, kind, name, targets in skeleton:
        if k >= cut:
            break
        if kind == "pad":
            continue
        qs = [q for q, _ in targets] if kind == "meas" else targets
        if kind == "two":
            for a, b in zip(qs[::2], qs[1::2]):
                parents[root(a)] = root(b)
        if name not in _CSS:
            seeds.update(qs)
    phase_roots = {root(q) for q in seeds}
    core = [q for q in range(circuit.num_qubits) if root(q) in phase_roots]
    if len(core) > 64:
        return None
    return {"cut": cut, "boundary": len(circuit.flattened()), "core_qubits": core, "queries": []}


def _cultivation_plan(circuit):
    """Return {cut, boundary, core_qubits, queries}, or None if unsupported.

    Each query is (global record index, qubit list, basis string, pivot, inverted).
    Original flattened instruction indices are retained, including frame noise.
    Terminal compute/uncompute must be an exact atomic palindrome, with basis
    rotations followed by distinct-control CXs into the readout pivot. This
    ensures that frozen lost operands simply restrict the recovered observable.
    """
    circuit = circuit.flattened()
    flat = []
    for i in range(len(circuit)):
        inst = circuit[i]
        assert isinstance(inst, stim.CircuitInstruction)
        flat.append(inst)
    boundary = next((i for i, inst in enumerate(flat) if inst.tag == "noiseless"
                     and inst.name not in _ANNOTATIONS | _FRAME), None)
    if boundary is None:
        return None
    phase, atoms, record = [], [], 0
    for k, inst in enumerate(flat):
        name = inst.name
        if name in _ANNOTATIONS | _FRAME:
            continue
        targets = inst.targets_copy()
        if any(not target.is_qubit_target for target in targets):
            return None
        qs = [target.value for target in targets]
        if k < boundary:
            if name not in _PHYSICAL:
                return None
            if name in {"S", "S_DAG"}:
                phase.append(k)
            if name in _MEASURE:
                record += len(qs)
        else:
            if inst.tag != "noiseless" or name not in {"H", "H_YZ", "CX", "M"}:
                return None
            width = 2 if name == "CX" else 1
            for j in range(0, len(qs), width):
                inverted = targets[j].is_inverted_result_target if name == "M" else False
                atoms.append((name, tuple(qs[j:j + width]), inverted, record if name == "M" else -1))
                record += name == "M"
    if not phase:
        return None
    cut = next((i for i in range(phase[-1] + 1, boundary) if flat[i].name == "CX"), boundary)
    parents = list(range(circuit.num_qubits))

    def root(q):
        while parents[q] != q:
            parents[q] = parents[parents[q]]
            q = parents[q]
        return q

    for inst in flat[:cut]:
        if inst.name == "CX":
            qs = [target.value for target in inst.targets_copy()]
            for a, b in zip(qs[::2], qs[1::2]):
                parents[root(a)] = root(b)
    phase_roots = {root(target.value) for k in phase for target in flat[k].targets_copy()}
    core = [q for q in range(circuit.num_qubits) if root(q) in phase_roots]
    if len(core) > 64:
        return None
    queries, pos, mixed = [], 0, 0
    while pos < len(atoms):
        end = next((j for j in range(pos, len(atoms)) if atoms[j][0] == "M"), None)
        if end is None:
            return None
        compute = atoms[pos:end]
        if atoms[end + 1:end + 1 + len(compute)] != compute[::-1]:
            return None
        _, (pivot,), inverted, index = atoms[end]
        rotations, controls, saw_cx = {}, [], False
        for name, qs, _, _ in compute:
            if name == "CX":
                a, b = qs
                if b != pivot or a == pivot or a in controls:
                    return None
                controls.append(a)
                saw_cx = True
            else:
                q = qs[0]
                if saw_cx or q in rotations:
                    return None
                rotations[q] = "X" if name == "H" else "Y"
        support = sorted([pivot, *controls])
        if not rotations.keys() <= set(support):
            return None
        bases = "".join(rotations.get(q, "Z") for q in support)
        mixed += len(set(bases)) != 1 or bases[0] == "Y"
        queries.append((index, support, bases, pivot, inverted))
        pos = end + 1 + len(compute)
    if mixed != 1:
        return None
    return {"cut": cut, "boundary": boundary, "core_qubits": core, "queries": queries}
