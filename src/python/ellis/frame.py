"""Masked Pauli-frame simulator for Clifford circuits with atom loss (gate-drop semantics).

A shot with loss runs its own Clifford circuit C_s = C_0 minus every operation on a qubit
after it is lost.  Stim-style frame simulation of C_s needs
  (i)  the frame propagated through C_s: every gate is AND-masked by per-shot "alive" bits;
  (ii) a reference sample of C_s: we reuse C_0's reference, which is exact whenever it is also
       a valid noiseless sample of every C_s (checked by check_exact.py).
A lost qubit is simply frozen: nothing touches its frame again, its readouts are heralded and
filled with a uniformly random placeholder bit (Clifft's convention).

Loss instruction: `I_ERROR[loss](p) q...` (ppvm spelling; Stim parses and ignores it).
Reload instruction: `I_ERROR[reload](p) q...` loads a fresh atom in |0> into each empty slot q with
probability p (default 1) and leaves a present atom alone; a failed reload leaves the slot empty, which is
the same as a successful reload followed by a loss right after the marker (how explicit loss patterns write
it).  It can stand anywhere; to reload at a reset, write the marker just before the reset (before R / RX /
RY, or between the readout and reset of a split MR / MRX / MRY), so the fresh atom is reset like any other.
`sample` below is the numpy reference implementation with all research options (gauges, reference
tables, ...).  `compile_sampler` returns the C++ engine (ellis._core, built on Stim's SIMD bit
tables): the same semantics in plain mode, batched and multithreaded, at close to Stim's speed.
"""
import numpy as np
import stim

ANNOTATIONS = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}
MEAS = {"M": "Z", "MZ": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRZ": "Z", "MRX": "X", "MRY": "Y"}
RESET = {"R": "Z", "RZ": "Z", "RX": "X", "RY": "Y"}


PAULI = {"X", "Y", "Z"}
UNITARY_1Q = {n for n, g in stim.gate_data().items() if g.is_unitary and g.is_single_qubit_gate}
UNITARY_2Q = {n for n, g in stim.gate_data().items() if g.is_unitary and g.is_two_qubit_gate}
PAULI_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "I_ERROR"}
SUPPORTED = ANNOTATIONS | set(MEAS) | set(RESET) | UNITARY_1Q | UNITARY_2Q | PAULI_NOISE


def _delta_plan(name):
    """Symplectic action of a unitary gate as XOR plans: for each frame bit (x0, z0, x1, z1, ...), the
    input bits whose XOR is its change (None if unchanged).  Read from Stim's own tableau, so every
    Clifford gate Stim knows is covered."""
    t = stim.gate_data(name).tableau
    assert t is not None, name
    n = len(t)
    images = [t.x_output(k) if b == 0 else t.z_output(k) for k in range(n) for b in (0, 1)]
    rows = [[v for q in range(n) for v in (p[q] in (1, 2), p[q] in (2, 3))] for p in images]
    plan = []
    for j in range(2 * n):
        ins = {i for i in range(2 * n) if rows[i][j]} ^ {j}     # new_j = old_j ^ XOR(ins)
        plan.append(tuple(sorted(ins)) or None)
    return tuple(plan)


_PLAN = {name: _delta_plan(name) for name in UNITARY_1Q | UNITARY_2Q}


def _xor(frames, ins):
    """XOR of the selected frame rows (the caller ANDs it with the alive mask, which allocates)."""
    if len(ins) == 1:
        return frames[ins[0]]
    out = frames[ins[0]] ^ frames[ins[1]]
    for i in ins[2:]:
        out ^= frames[i]
    return out


def _distinct_chunks(qs, width):
    """Split a target list into consecutive chunks (of whole `width`-tuples) without a repeated qubit, so
    that vectorized updates `x[qs] ^= ...` act once per target, as Stim does for `H 0 0`."""
    chunks, cur, seen = [], [], set()
    for i in range(0, len(qs), width):
        g = qs[i:i + width]
        if seen.intersection(g):
            chunks.append(cur); cur, seen = [], set()
        cur += g; seen.update(g)
    chunks.append(cur)
    return chunks


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
    out = stim.Circuit()
    for inst in circuit.flattened():
        if not (is_loss(inst) or is_reload(inst)) and inst.name not in PAULI:
            out.append(inst)
    return out


ONE_Q_CLIFFORD = UNITARY_1Q                # every single-qubit Clifford Stim knows (incl. I and the Paulis)
_CANON = {}
for _g in ("I", "H", "S", "SQRT_X_DAG", "C_XYZ", "C_ZYX"):          # the all-'+' representative
    _t = stim.Tableau.from_named_gate(_g)
    _CANON[(str(_t.x_output(0))[1:], str(_t.z_output(0))[1:])] = _g


def _split(u: stim.Tableau):
    """u = P * C with C the canonical (sign-free) representative and P a Pauli."""
    c = _CANON[(str(u.x_output(0))[1:], str(u.z_output(0))[1:])]
    p = u * stim.Tableau.from_named_gate(c).inverse()
    sx, sz = str(p.x_output(0))[0] == "-", str(p.z_output(0))[0] == "-"
    return c, {(False, False): None, (True, False): "Z", (False, True): "X", (True, True): "Y"}[(sx, sz)]


def canonical_reference(circuit: stim.Circuit):
    """Reference circuit in which every run of single-qubit Cliffords is replaced by its sign-free
    representative C; the leftover Pauli P (run = P*C) goes into the frame instead.
    Returns (reference_circuit, {instruction_index: [(qubit, pauli)] applied before that index})."""
    ref, inserts, runs = stim.Circuit(), {}, {}
    def close(q, k):
        if q in runs:
            c, p = _split(stim.Tableau.from_circuit(stim.Circuit("\n".join(f"{g} 0" for g in runs.pop(q)))))
            if c != "I": ref.append(c, [q])
            if p: inserts.setdefault(k, []).append((q, p))
    flat = list(circuit.flattened())
    for k, inst in enumerate(flat):
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in ONE_Q_CLIFFORD:
            for q in qs: runs.setdefault(q, []).append(inst.name)
            continue
        if not (stim.gate_data(inst.name).is_noisy_gate and not stim.gate_data(inst.name).produces_measurements) \
                or is_loss(inst) or is_reload(inst):
            for q in qs: close(q, k)
        if not (is_loss(inst) or is_reload(inst)):
            ref.append(inst)
    for q in list(runs): close(q, len(flat))
    return ref, inserts


def compile_sampler(circuit: stim.Circuit, *, reference=None, forced_losses=None,
                    seed: int | None = None, batch_size: int = 0, simd_width: int = 0):
    """C++ masked-frame sampler for `circuit` (plain mode: one reference record, optional forced losses).
    .sample(shots) -> (measurements, heralds) like `sample`; .sample_detectors(shots) -> (detectors, observables).
    Both take bit_packed=, heralds= and threads=; see ellis._core.MaskFrameSampler."""
    from ellis._core import MaskFrameSampler
    return MaskFrameSampler(circuit, reference=reference, forced_losses=forced_losses, seed=seed,
                            batch_size=batch_size, simd_width=simd_width)


def sample(circuit: stim.Circuit, shots: int, seed: int = 0, reference=None,
           canonical: bool = False, gauge=None, per_shot_inserts=None, deterministic: bool = False,
           forced_losses=None, ref_table=None, reference_blocks=None, reference_provider=None,
           return_losses: bool = False, software_feedforward: bool = False, lost_partner_idle_p: float = 0.0):
    """Returns (measurements, heralds) as (shots, num_measurements) bool arrays.
    return_losses: plain mode only; also return each shot's loss events as {instruction_index: [qubits]}, one dict
           per shot, recording only atoms that were alive (a repeated loss of a lost atom is not an event), plus
           each failed reload as a loss right after its marker.
    software_feedforward: a CX tagged 'feedforward_t' / 'feedforward_c' (a classically controlled Pauli done in software,
           e.g. SWAP SE's correction) acts whenever its corrected operand (target / control) is present, even if the measured
           partner is lost: the correction then uses the lost atom's frozen frame, as a Stim feedback gate `CX rec[-k] q`
           whose control record is masked only afterwards would.  Default: an ordinary gate, dropped if either is lost.
    lost_partner_idle_p: optional depolarizing probability on a present operand of a CX/CZ whose partner is lost
           (Liu et al., App. M1).  Software feed-forward does not represent a physical idle location. Default: zero.
    gauge: {instruction_index: [(qubit, 'X'|'Z'|'Y')]} extra frame Paulis applied (masked) BEFORE that
           instruction in every shot (a solved "sign ownership" gauge, see gauge_ref.py).
    per_shot_inserts / deterministic: used by the solver to measure each insert's flip pattern:
           shot j gets only per_shot_inserts[j] = (index, qubit, pauli); no randomness at all.
    ref_table: (key_atoms, {frozenset(lost key atoms): reference}) from ref_table.py, for circuits with
           no single valid reference (plain mode only).  Each shot takes the reference of its
           class; a shot whose class is not in the table takes the exact reference of its own lossy
           circuit (one Stim reference sample per such shot, the frame is kept).
           Then returns (measurements, heralds, fell_back).
    reference_blocks: experimental block_reference.compile_blocks() result for this exact circuit. XORs the active blocks' reference deltas; not certified exact.
    reference_provider: an exact get_many(patterns) provider for this circuit.
           Captures each shot's actual new losses and supplies its own valid reference record."""
    assert ref_table is None or not (canonical or gauge), "ref_table: plain mode"
    if reference_blocks is not None:
        assert ref_table is None and not (canonical or gauge), "reference_blocks: plain mode only"
        assert reference_blocks.circuit == circuit.flattened()
    if reference_provider is not None:
        assert ref_table is None and reference_blocks is None and not (canonical or gauge), "reference_provider: plain mode only"
        assert reference_provider.circuit == circuit.flattened()
    if return_losses:
        assert ref_table is None and reference_blocks is None and reference_provider is None, "return_losses: plain mode only"
    track = return_losses or ref_table is not None or reference_blocks is not None or reference_provider is not None
    validate(circuit)
    rng = np.random.default_rng(seed)
    inserts = {}
    if canonical:
        ref_circ, inserts = canonical_reference(circuit)
        if reference is None:
            reference = ref_circ.reference_sample()
    elif reference is None:
        reference = strip_loss(circuit).reference_sample()
    if reference is None:
        reference = np.zeros(circuit.num_measurements, bool)
    for k, lst in (gauge or {}).items():
        inserts.setdefault(k, []).extend(lst)
    shot_ins = {}
    for j, (k, q, p, *cond) in enumerate(per_shot_inserts or ()):
        shot_ins.setdefault(k, []).append((j, q, p, tuple(cond[0]) if cond else (q,)))
    forced_losses = forced_losses or {}
    n = circuit.num_qubits
    x = np.zeros((n, shots), bool)
    rnd = (lambda k: np.zeros((k, shots), bool)) if deterministic else (lambda k: rng.random((k, shots)) < 0.5)
    z = rnd(n)                                # |0> start: random Z stabilizers
    alive = np.ones((n, shots), bool)
    events = []                               # (index, qubit, shots newly lost there), for the fallback
    rec, her = [], []
    coin = lambda p, k: rng.random((k, shots)) < p

    def reset(qs, basis):
        ok = alive[qs]
        r = rnd(len(qs))
        nx = {"Z": np.zeros_like(r), "X": r, "Y": r}[basis]
        nz = {"Z": r, "X": np.zeros_like(r), "Y": r}[basis]
        x[qs] = np.where(ok, nx, x[qs])
        z[qs] = np.where(ok, nz, z[qs])

    for k, inst in enumerate(circuit.flattened()):
        for q, p, *cond in inserts.get(k, ()):   # canonical-run / gauge Paulis, masked by the
            on = alive[q].copy()                  # condition qubits (default: the qubit itself)
            for cq in (cond[0] if cond else ()): on &= alive[cq]
            if p != "Z": x[q] ^= on
            if p != "X": z[q] ^= on
        for j, q, p, cond in shot_ins.get(k, ()):
            on = all(alive[cq, j] for cq in cond)
            if p != "Z": x[q, j] ^= on
            if p != "X": z[q, j] ^= on
        if k - 1 in forced_losses:              # forced loss right after original instruction k-1
            if track:
                events += [(k - 1, q, alive[q].copy()) for q in forced_losses[k - 1]]
            alive[forced_losses[k - 1]] = False
        name, args = inst.name, inst.gate_args_copy()
        if name in ANNOTATIONS:
            if name == "MPAD":                   # its value lives in the reference record; the frame adds 0
                n_pad = len(inst.targets_copy())
                rec.append(np.zeros((n_pad, shots), bool)); her.append(np.zeros((n_pad, shots), bool))
            continue
        # Measurement inversions (`M !q`) need nothing here: they live in the reference record.
        width = 2 if name in UNITARY_2Q or name == "DEPOLARIZE2" else 1
        for qs in _distinct_chunks([t.value for t in inst.targets_copy()], width):
            if name == "I_ERROR":
                if inst.tag == "loss":
                    hit = coin(args[0], len(qs))
                    if track:
                        events += [(k, q, hit[i] & alive[q]) for i, q in enumerate(qs)]
                    alive[qs] &= ~hit
                elif inst.tag == "reload":     # a fresh |0> in each empty slot: X part 0, Z part a fresh gauge bit
                    empty = ~alive[qs]
                    if args and args[0] < 1:
                        failed = empty & ~coin(args[0], len(qs))
                        if track:
                            events += [(k, q, failed[i]) for i, q in enumerate(qs)]
                        empty &= ~failed
                    x[qs] &= ~empty
                    z[qs] = np.where(empty, rnd(len(qs)), z[qs])
                    alive[qs] |= empty
            elif name in RESET:
                reset(qs, RESET[name])
            elif name in PAULI:                  # deterministic Pauli "error", masked
                if not canonical:                # (canonical mode folded it into the run's Pauli insert)
                    a = alive[qs]
                    if name != "Z": x[qs] ^= a
                    if name != "X": z[qs] ^= a
            elif name in UNITARY_1Q:             # symplectic action only (signs live in reference/inserts)
                a, old = alive[qs], (x[qs], z[qs])
                dx, dz = (_xor(old, ins) & a if ins else None for ins in _PLAN[name])
                if dx is not None: x[qs] ^= dx
                if dz is not None: z[qs] ^= dz
            elif name in UNITARY_2Q:             # whole gate dropped if either operand is lost
                plan = _PLAN[name]
                sw = software_feedforward and inst.tag in ("feedforward_t", "feedforward_c")
                for c, t in zip(qs[::2], qs[1::2]):
                    a = (alive[t] if inst.tag == "feedforward_t" else alive[c]) if sw else alive[c] & alive[t]
                    rows = (x[c], z[c], x[t], z[t])
                    deltas = [_xor(rows, ins) & a if ins else None for ins in plan]
                    for row, d in zip(rows, deltas):
                        if d is not None: row ^= d
                    if lost_partner_idle_p and name in ("CX", "CZ") and not inst.tag.startswith("feedforward"):
                        for q, partner in ((c, t), (t, c)):
                            idle = alive[q] & ~alive[partner]
                            if idle.any():
                                hit = (rng.random(shots) < lost_partner_idle_p) & idle
                                pauli = rng.integers(1, 4, shots)
                                x[q] ^= hit & (pauli != 3)
                                z[q] ^= hit & (pauli != 1)
            elif name in MEAS:
                basis = MEAS[name]
                flip = coin(args[0], len(qs)) if args else np.zeros((len(qs), shots), bool)
                raw = {"Z": x[qs], "X": z[qs], "Y": x[qs] ^ z[qs]}[basis]
                lost = ~alive[qs]
                rec.append(np.where(lost, rnd(len(qs)), raw ^ flip))
                her.append(lost)
                r = rnd(len(qs)) & ~lost           # collapse: randomize the measured-basis stabilizer
                if basis == "Z": z[qs] ^= r
                elif basis == "X": x[qs] ^= r
                else: x[qs] ^= r; z[qs] ^= r
                if name.startswith("MR"):
                    reset(qs, basis)
            elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                e = coin(args[0], len(qs))
                if name != "Z_ERROR": x[qs] ^= e
                if name != "X_ERROR": z[qs] ^= e
            elif name == "DEPOLARIZE1":
                hit = coin(args[0], len(qs)); p = rng.integers(1, 4, (len(qs), shots))
                x[qs] ^= hit & (p != 3); z[qs] ^= hit & (p != 1)
            elif name == "DEPOLARIZE2":
                a, b = qs[::2], qs[1::2]
                hit = coin(args[0], len(a)) & alive[a] & alive[b]   # pair noise dropped if a partner is lost
                p = rng.integers(1, 16, (len(a), shots))
                x[a] ^= hit & ((p & 1) > 0); z[a] ^= hit & ((p & 2) > 0)
                x[b] ^= hit & ((p & 4) > 0); z[b] ^= hit & ((p & 8) > 0)
            else:
                raise NotImplementedError(f"{name} (see frame.validate)")
    rec = np.concatenate(rec).T
    m = rec ^ reference[None, :]
    if reference_blocks is not None or reference_provider is not None:
        patterns = [{} for _ in range(shots)]
        for k, q, newly in events:
            for j in np.flatnonzero(newly):
                patterns[j].setdefault(k, []).append(q)
        if reference_provider is not None:
            m = rec ^ reference_provider.get_many(patterns)
        else:
            for j, pattern in enumerate(patterns):
                m[j] = rec[j] ^ reference_blocks.predict(pattern)
        return m, np.concatenate(her).T
    if return_losses:
        patterns = [{} for _ in range(shots)]
        for k, q, newly in events:
            for j in np.flatnonzero(newly):
                patterns[j].setdefault(k, []).append(q)
        return m, np.concatenate(her).T, patterns
    if ref_table is None:
        return m, np.concatenate(her).T
    atoms, refs = np.array(ref_table[0]), ref_table[1]
    fell, skel = np.zeros(shots, bool), None
    codes, inv = np.unique(np.packbits(~alive[atoms], axis=0).T, axis=0, return_inverse=True)
    inv = inv.ravel()
    for u, code in enumerate(codes):
        idx = np.flatnonzero(inv == u)
        cls = frozenset(atoms[np.unpackbits(code)[:len(atoms)].astype(bool)].tolist())
        if cls in refs:
            m[idx] = rec[idx] ^ refs[cls][None, :]
            continue
        fell[idx] = True
        skel = skel or reference_skeleton(circuit)
        for j in idx:                         # exact reference of this shot's own lossy circuit
            losses = {}
            for kk, q, newly in events:
                if newly[j]:
                    losses.setdefault(kk, []).append(q)
            m[j] = rec[j] ^ lossy_reference(skel, losses)
    return m, np.concatenate(her).T, fell


def reference_skeleton(circuit: stim.Circuit, software_feedforward: bool = False):
    """The parts of a circuit a plain-mode reference depends on: gates, resets, measurements and MPAD
    (noise, loss, annotations and Pauli gates dropped), as (index, kind, name, qubits).  With software_feedforward,
    feed-forward CXs get kind 'fft' / 'ffc' (kept while the corrected target / control is present; see sample)."""
    out = []
    for k, inst in enumerate(circuit.flattened()):
        name, gd, targets = inst.name, stim.gate_data(inst.name), inst.targets_copy()
        if name == "MPAD":
            out.append((k, "pad", name, [t.value for t in targets]))
        elif is_reload(inst):
            out.append((k, "reload", "R", [t.value for t in targets]))
        elif name in ANNOTATIONS or name in PAULI or (gd.is_noisy_gate and not gd.produces_measurements):
            continue
        elif name in MEAS:                    # keep inverted targets (`M !q`)
            out.append((k, "meas", name, [(t.value, t.is_inverted_result_target) for t in targets]))
        else:
            kind = "two" if gd.is_two_qubit_gate else "one"
            if software_feedforward and inst.tag in ("feedforward_t", "feedforward_c"):
                kind = "fft" if inst.tag == "feedforward_t" else "ffc"
            out.append((k, kind, name, [t.value for t in targets]))
    return out


def lossy_reference(skeleton, losses):
    """Reference sample of the lossy circuit C_s with Pauli gates stripped (what the masked frame
    needs for loss pattern s), built as text; a lost readout becomes MPAD 0."""
    pending, i, lost, lines = sorted(losses.items()), 0, set(), []
    for k, kind, name, qs in skeleton:
        while i < len(pending) and pending[i][0] < k:
            lost.update(pending[i][1]); i += 1
        if kind == "pad":
            lines.append("MPAD " + " ".join(map(str, qs)))
        elif kind == "reload":
            for q in qs:
                if q in lost:
                    lost.discard(q)
                    lines.append(f"R {q}")
        elif kind == "meas":
            for q, inv in qs:
                if q not in lost:
                    lines.append(f"{name} {'!' * inv}{q}")
                else:
                    lines.append("MPAD 0")
        else:
            pairs = zip(qs[::2], qs[1::2]) if kind in ("two", "fft", "ffc") else ((q,) for q in qs)
            if kind == "fft":
                keep = [str(q) for g in pairs if g[1] not in lost for q in g]
            elif kind == "ffc":
                keep = [str(q) for g in pairs if g[0] not in lost for q in g]
            else:
                keep = [str(q) for g in pairs if not lost.intersection(g) for q in g]
            if keep:
                lines.append(f"{name} " + " ".join(keep))
    return stim.Circuit("\n".join(lines)).reference_sample()


def explicit_lossy_circuit(circuit: stim.Circuit, losses, software_feedforward: bool = False) -> stim.Circuit:
    """C_s for one loss pattern, as an ordinary Stim circuit.
    losses = {instruction_index: [qubits lost right after that instruction]}; index -1: lost from the start.
    Operations on lost qubits are removed; a lost readout becomes a fresh |+> ancilla measured
    in Z (uniform placeholder bit), so the measurement record keeps its layout.  A reload marker on an
    empty slot becomes `R q`."""
    anc = circuit.num_qubits
    out = stim.Circuit()
    lost = set(losses.get(-1, ()))
    for k, inst in enumerate(circuit.flattened()):
        name, targets = inst.name, inst.targets_copy()
        qs = [t.value for t in targets]
        if is_reload(inst):
            for q in qs:
                if q in lost:
                    lost.discard(q)
                    out.append("R", [q])
        elif name in ANNOTATIONS or is_loss(inst):
            if name != "I_ERROR":
                out.append(inst)
        elif name in MEAS:
            for t in targets:
                q = t.value
                if q in lost:
                    out.append("RX", [anc]); out.append("M", [anc])
                else:
                    out.append(name, [t], inst.gate_args_copy())     # keeps an inverted target `!q`
        else:
            two = stim.gate_data(name).is_two_qubit_gate
            groups = list(zip(qs[::2], qs[1::2])) if two else [(q,) for q in qs]
            if software_feedforward and inst.tag in ("feedforward_t", "feedforward_c"):    # see sample()
                i = 1 if inst.tag == "feedforward_t" else 0
                keep = [q for g in groups if g[i] not in lost for q in g]
            else:
                keep = [q for g in groups if not (set(g) & lost) for q in g]
            if keep:
                out.append(name, keep, inst.gate_args_copy())
        lost |= set(losses.get(k, ()))
    return out
