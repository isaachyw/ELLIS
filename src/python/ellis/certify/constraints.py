"""Loss slots and families, exact sign constraints of lossy circuits, and the GF(2) search for a reference or
a contradiction witness (paper section 3, Theorem 1)."""

import numpy as np
import stim

from ellis import frame as mf

PLACEMENTS = ("2q", "any")  # loss_slots; certify() also takes "anywhere"


# ------------------------------------------------------------------ loss slots and families
def loss_slots(circuit: stim.Circuit, placement: str = "2q", include_noiseless: bool = False):
    """(instruction index k, qubit q): q can be lost right after instruction k.
    placement '2q': after each two-qubit gate, for both operands (the benchmark noise model);
    'any': after every gate, reset or measurement.  Instructions tagged `noiseless` are skipped unless
    include_noiseless."""
    assert placement in PLACEMENTS, placement
    slots = []
    for k, inst in enumerate(circuit.flattened()):
        name, gd = inst.name, stim.gate_data(inst.name)
        if name in mf.ANNOTATIONS or (gd.is_noisy_gate and not gd.produces_measurements):
            continue
        if inst.tag == "noiseless" and not include_noiseless:
            continue
        if placement == "2q" and name not in mf.UNITARY_2Q:
            continue
        slots += [(k, q) for q in dict.fromkeys(t.value for t in inst.targets_copy() if t.is_qubit_target)]
    return slots


def _events(circuit: stim.Circuit):
    """Per qubit, the instruction indices a loss can act through: two-qubit gates, readouts, reload markers."""
    ev = {}
    for k, inst in enumerate(circuit.flattened()):
        name = inst.name
        if name in mf.UNITARY_2Q or name in mf.MEAS or mf.is_reload(inst):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    ev.setdefault(t.value, []).append(k)
    return ev


def representative_slots(circuit: stim.Circuit, slots):
    """One slot per ghost-equivalence class: (k, q) and (k', q) are equivalent when no event of q lies in
    between, because then the same later operations of q are dropped.  Slots after q's last event (losing
    an atom that does nothing afterwards) change nothing and are removed."""
    ev, seen, reps = _events(circuit), set(), []
    for k, q in slots:
        nxt = next((e for e in ev.get(q, ()) if e > k), None)
        if nxt is None or (q, nxt) in seen:
            continue
        seen.add((q, nxt))
        reps.append((k, q))
    return reps


def loss_family(
    circuit: stim.Circuit,
    placement: str = "2q",
    include_noiseless: bool = False,
    pairs: bool = False,
    n_random: int = 1000,
    max_losses: int = 4,
    seed: int = 0,
    slots=None,
):
    """Loss patterns {instruction index: [qubits lost right after it]}: no loss, every single loss
    (one per ghost class), optionally every pair, then n_random patterns of 2..max_losses losses.
    slots: the loss locations, default loss_slots(circuit, placement, include_noiseless)."""
    if slots is None:
        slots = loss_slots(circuit, placement, include_noiseless)
    reps = representative_slots(circuit, slots)
    fam = [{}] + [{k: [q]} for k, q in reps]
    if pairs:
        fam += [_pattern((a, b)) for i, a in enumerate(reps) for b in reps[i + 1 :] if a[1] != b[1]]
    rng = np.random.default_rng(seed)
    for _ in range(n_random if len(reps) > 1 else 0):
        k = int(rng.integers(2, max(2, max_losses) + 1))
        pick, seen = [], set()
        for i in rng.permutation(len(reps)):
            if reps[i][1] not in seen:
                seen.add(reps[i][1])
                pick.append(reps[i])
            if len(pick) == k:
                break
        fam.append(_pattern(pick))
    return fam


def _pattern(slots):
    out = {}
    for k, q in slots:
        out.setdefault(k, []).append(q)
    return out


# ------------------------------------------------------------------ exact sign constraints
def _noiseless(circuit: stim.Circuit) -> stim.Circuit:
    """Drop noise channels and measurement-flip probabilities (keeps inverted targets and MPAD)."""
    # Appending (name, targets_copy(), args) makes Stim probe every target with a throwing pybind11 cast, which is slow
    # where C++ unwinding is (macOS).  Append the instruction itself, or its text, instead.
    lines = []
    for inst in circuit.flattened():
        assert isinstance(inst, stim.CircuitInstruction)  # flattened: no REPEAT blocks
        gd = stim.gate_data(inst.name)
        if gd.is_noisy_gate and not gd.produces_measurements:
            continue
        if not gd.produces_measurements or not inst.gate_args_copy():
            lines.append(str(inst))
        else:
            tag = f"[{inst.tag}]" if inst.tag else ""
            lines.append(f"{inst.name}{tag} {str(inst).split(' ', 1)[1]}")
    return stim.Circuit("\n".join(lines))


def _one_target_per_measure_reset(circuit: stim.Circuit) -> stim.Circuit:
    """Stim 1.16's flow_generators reads `MR q q` (one instruction measuring and resetting q twice) as if both
    readouts came before one reset, reporting rec[0] xor rec[1] as deterministic; its simulator, correctly, makes
    the second readout deterministic instead.  A TICK between single-target measure-resets keeps them apart."""
    lines = []
    for inst in circuit:
        if inst.name.startswith("MR"):
            head, *targets = str(inst).split(" ")
            lines += [f"TICK\n{head} {t}" for t in targets]
        else:
            lines.append(str(inst))
    return stim.Circuit("\n".join(lines))


def constraints(circuit: stim.Circuit, losses):
    """Exact sign constraints of the lossy circuit C_s: a basis of its deterministic parities y (as int
    bitmasks over measurement indices) with their values sigma_s(y), for the Pauli-stripped reference
    circuit the masked frame uses.  From Stim's flow generators (no sampling): the parities are the flows
    with no X/Y on the input (every qubit starts in |0>) and no output Pauli; the values come from one
    noiseless record of C_s."""
    # rebuild first (loss patterns are keyed by the circuit's own instruction indices), then strip
    cs = _one_target_per_measure_reset(_noiseless(mf.strip_loss(mf.explicit_lossy_circuit(circuit, losses))))
    n = cs.num_qubits
    ref_int = _as_int(cs.reference_sample())
    piv, rows = {}, []
    for g in cs.flow_generators():
        key = 0
        inp, out = g.input_copy(), g.output_copy()
        if len(inp):
            key |= _bits(inp.to_numpy()[0], 0)
        if len(out):
            ox, oz = out.to_numpy()
            key |= _bits(ox, n) | _bits(oz, 2 * n)
        y = sum(1 << int(r) for r in g.measurements_copy())
        while key:
            h = key.bit_length() - 1
            if h not in piv:
                piv[h] = (key, y)
                break
            pk, py = piv[h]
            key ^= pk
            y ^= py
        else:
            if y:
                rows.append((y, (y & ref_int).bit_count() & 1))
    return rows


def _bits(arr, offset):
    return sum(1 << (offset + int(q)) for q in np.flatnonzero(arr))


def _as_int(record) -> int:
    return sum(1 << int(i) for i in np.flatnonzero(record))  # Python ints: records exceed 64 bits


def check_reference(circuit: stim.Circuit, reference, family):
    """Loss patterns of the family for which the plain masked frame with this reference is NOT exact."""
    r = _as_int(reference)
    return [s for s in family if any(((y & r).bit_count() & 1) != v for y, v in constraints(circuit, s))]


# ------------------------------------------------------------------ GF(2) system with provenance
class _System:
    """Incremental GF(2) system y.r = v.  Each pivot keeps the tags of the input rows it is a sum of (an int
    bitmask), so a contradiction 0 = 1 comes with the tags that produce it: exactly (XOR) when tags are row
    ids, or a covering set (OR) when tags are loss-pattern indices, as in the search (one bit per pattern
    keeps masks small for millions of rows; two rows of one pattern may cancel, so XOR would lose it)."""

    def __init__(self, cover: bool = False):
        self.piv, self.cover = {}, cover

    def add(self, y, v, tag):
        prov = 1 << tag
        while y:
            h = y.bit_length() - 1
            if h not in self.piv:
                self.piv[h] = (y, v, prov)
                return None
            py, pv, pp = self.piv[h]
            y ^= py
            v ^= pv
            prov = (prov | pp) if self.cover else (prov ^ pp)
        return prov if v else None

    def solve(self, default: int = 0) -> int:
        r = default
        for h in sorted(self.piv):  # ascending: the other bits of y (all < h) are fixed
            y, v, _ = self.piv[h]
            want = (((y & ~(1 << h)) & r).bit_count() & 1) ^ v
            r = (r | (1 << h)) if want else (r & ~(1 << h))
        return r


def _contradiction(rows_by_context):
    """Rows (context index, y, v) whose sum is 0 = 1, or None if the system is consistent."""
    sysm, flat = _System(), []
    for ci, rows in rows_by_context:
        for y, v in rows:
            flat.append((ci, y, v))
            if (prov := sysm.add(y, v, len(flat) - 1)) is not None:
                return [flat[i] for i in range(len(flat)) if (prov >> i) & 1]
    return None


_WORKER = {}


def _init_worker(text: str):
    _WORKER["circuit"] = stim.Circuit(text)


def _worker_constraints(losses):
    return constraints(_WORKER["circuit"], losses)


def _constraint_stream(circuit: stim.Circuit, family, workers: int):
    """constraints() of each loss pattern, in family order, from `workers` processes when workers > 1."""
    if workers <= 1:
        yield from (constraints(circuit, s) for s in family)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(circuit),)) as ex:
        try:
            yield from ex.map(_worker_constraints, family, chunksize=max(1, min(16, len(family) // (4 * workers))))
        finally:
            ex.shutdown(cancel_futures=True)


class _Search:
    """Incremental exact search: the GF(2) system of the sign constraints of every loss pattern added so far."""

    def __init__(self, circuit: stim.Circuit, workers: int = 1):
        self.circuit, self.workers = circuit, workers
        self.family, self.sys = [], _System(cover=True)

    def extend(self, patterns):
        """Add loss patterns; returns None, or a minimal witness as soon as the system contradicts itself."""
        start = len(self.family)
        self.family += list(patterns)
        stream = _constraint_stream(self.circuit, self.family[start:], self.workers)
        try:
            for i, rows in enumerate(stream, start):
                for y, v in rows:
                    if (prov := self.sys.add(y, v, i)) is not None:
                        ctx = [j for j in range(i + 1) if (prov >> j) & 1]
                        return _minimize(ctx, self._rows_cached(), self.family)
        finally:
            stream.close()
        return None

    def _rows_cached(self):
        cache = {}

        def rows_of(j):
            if j not in cache:
                cache[j] = constraints(self.circuit, self.family[j])
            return cache[j]

        return rows_of

    def reference(self) -> np.ndarray:
        """A record satisfying every constraint so far (Stim's loss-free reference on the free bits)."""
        r = self.sys.solve(_as_int(mf.strip_loss(_noiseless(self.circuit)).reference_sample()))
        return np.array([(r >> j) & 1 for j in range(self.circuit.num_measurements)], bool)


def search(circuit: stim.Circuit, family, workers: int = 1):
    """Exact GF(2) search over a family of loss patterns; the constraints of the patterns are computed by
    `workers` processes.  Returns (reference fitting every pattern, None) or (None, minimal witness)."""
    srch = _Search(circuit, workers)
    witness = srch.extend(family)
    return (None, witness) if witness is not None else (srch.reference(), None)


def verify_witness(circuit: stim.Circuit, witness, shots: int = 256) -> None:
    """Independent check of a witness with Stim sampling (not the flow code): each parity is constant, equal
    to its value, in its own lossy circuit (Pauli-stripped, noiseless); the parities cancel; the values sum
    to 1.  Raises AssertionError otherwise."""
    total = np.zeros(circuit.num_measurements, bool)
    for pattern, idx, val in witness:
        cs = _noiseless(mf.strip_loss(mf.explicit_lossy_circuit(circuit, pattern)))
        parity = np.bitwise_xor.reduce(cs.compile_sampler(seed=1).sample(shots)[:, idx], axis=1) if idx else np.zeros(shots, bool)
        assert (parity == bool(val)).all(), f"witness parity is not fixed to {val} under {pattern}"
        total[idx] ^= True
    assert not total.any() and sum(v for _, _, v in witness) % 2 == 1, "witness does not sum to 0 = 1"


def _minimize(ctx, rows, family):
    """Drop loss patterns while the rest still contradicts; return the witness rows grouped by pattern."""
    i = 0
    while i < len(ctx):
        trial = ctx[:i] + ctx[i + 1 :]
        if _contradiction([(c, rows(c)) for c in trial]) is not None:
            ctx = trial
        else:
            i += 1
    combo = _contradiction([(c, rows(c)) for c in ctx])
    out = []
    for c in ctx:
        y = v = 0
        for ci, yy, vv in combo:
            if ci == c:
                y ^= yy
                v ^= vv
        out.append((family[c], [j for j in range(y.bit_length()) if (y >> j) & 1], v))
    return out
