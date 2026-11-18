"""Independent oracles for `ellis.certify.frame_css`, built only from the paper's definition of frame-CSS circuits
(docs/fcss-prove/common_reference.tex, sec. "Frame-CSS circuits") and never from frame_css.py's algorithm.

O1  `brute_force_frame_css`: backtracking search over the 24 single-qubit Clifford frames of every segment.
O2  `common_reference_failures`: the loss patterns whose exact sign constraints (`constraints`) a reference violates.
O3  `random_frame_css`: circuits that are frame-CSS by construction; plus random Clifford circuits, near-miss
    mutations and verdict-preserving rewrites for the property tests.

Conventions.  The analysis runs on the Pauli-free, noiseless part of the circuit (Pauli gates and noise live in the
masked frame).  A frame F relates |psi_orig> = F |psi_local>, a single-qubit gate g maps F to gF, and the local
image of a physical Pauli P is F^dag P F.  Loss locations come from `loss_locations` (one per instruction, never
between two gates of one instruction); the prefix is every instruction up to and including the first of them.
Gates inside one instruction act one after the other, each in the frames of that moment.
"""
import itertools
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import stim
from ellis.certify import LossModel, constraints, loss_family, loss_locations

ONE_Q = ("I", "X", "Y", "Z", "H", "H_XY", "H_YZ", "H_NXY", "H_NXZ", "H_NYZ", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
         "SQRT_Y", "SQRT_Y_DAG", "C_XYZ", "C_ZYX", "C_NXYZ", "C_NZYX", "C_XNYZ", "C_XYNZ", "C_ZNYX", "C_ZYNX")
TWO_Q = tuple(sorted(n for n, g in stim.gate_data().items() if g.is_unitary and g.is_two_qubit_gate))
PAULI_GATES = {"X", "Y", "Z"}
RESET_BASIS = {"R": "Z", "RX": "X", "RY": "Y"}
MEAS_BASIS = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
ANNOTATIONS = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
PAULI_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
               "E", "ELSE_CORRELATED_ERROR", "I_ERROR", "II_ERROR"}
INERT = PAULI_GATES | PAULI_NOISE | ANNOTATIONS

# ------------------------------------------------------------------ the single-qubit Clifford group mod phase
TABLEAU = [stim.Tableau.from_named_gate(g) for g in ONE_Q]
_INDEX = {(str(t.x_output(0)), str(t.z_output(0))): i for i, t in enumerate(TABLEAU)}


def clifford_index(t: stim.Tableau) -> int:
    return _INDEX[(str(t.x_output(0)), str(t.z_output(0)))]


ID = clifford_index(stim.Tableau(1))
GATE = {g: clifford_index(stim.Tableau.from_named_gate(g)) for g in ONE_Q}
MUL = [[clifford_index(a * b) for b in TABLEAU] for a in TABLEAU]   # MUL[g][f]: the frame gF
INV = [clifford_index(t.inverse()) for t in TABLEAU]
LOCAL = [{p: (int(q.sign.real), "_XYZ"[q[0]]) for p in "XYZ" for q in [t.inverse()(stim.PauliString(p))]}
         for t in TABLEAU]                                          # LOCAL[f][P] = (sign, axis) of F^dag P F
ALL = (1 << 24) - 1


def mask(pred) -> int:
    return sum(1 << f for f in range(24) if pred(f))


def prep_ok(f: int, basis: str) -> bool:
    """The + eigenstate of `basis` is a + eigenstate of a channel in frame f."""
    return LOCAL[f][basis] in ((1, "X"), (1, "Z"))


def reads_channel(f: int, basis: str) -> bool:
    return LOCAL[f][basis][1] in "XZ"


def _pure(p: stim.PauliString, allowed) -> bool:
    return p.sign == 1 and all(v in allowed for v in p)


@cache
def good_pairs(name: str) -> tuple[int, ...]:
    """good_pairs(G)[fa]: bitmask of the fb for which W^dag G W is positive CSS, W = F_fa (x) F_fb."""
    g = stim.Tableau.from_named_gate(name)
    rows = []
    for fa in range(24):
        row = 0
        for fb in range(24):
            w = TABLEAU[fa] + TABLEAU[fb]
            local = w.inverse() * g * w
            if all(_pure(local.x_output(k), (0, 1)) and _pure(local.z_output(k), (0, 3)) for k in range(2)):
                row |= 1 << fb
        rows.append(row)
    return tuple(rows)


CSS_ABLE_2Q = tuple(g for g in TWO_Q if any(good_pairs(g)) and g != "II")


# ------------------------------------------------------------------ stabilizer groups as lists of stim.PauliString
def _has(p: stim.PauliString, q: int, part: str) -> bool:
    return p[q] in ((1, 2) if part == "x" else (2, 3))


def _eliminate(rows, cols):
    """Generators of the elements of <rows> with no bit in `cols` ((qubit, "x" | "z") pairs)."""
    rows = list(rows)
    for q, part in cols:
        piv = next((r for r in rows if _has(r, q, part)), None)
        if piv is not None:
            rows = [r * piv if _has(r, q, part) else r for r in rows if r is not piv]
    return [r for r in rows if r.weight]


def _rank(rows) -> int:
    piv = {}
    for r in rows:
        xs, zs = r.to_numpy()
        v = sum(1 << int(i) for i in np.flatnonzero(np.concatenate([xs, zs])))
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = v
                break
            v ^= piv[h]
    return len(piv)


def restrict(gens, keep, n: int):
    """Generators of the elements of <gens> supported on the qubits `keep` (the reduced state's stabilizers)."""
    return _eliminate(gens, [(q, part) for q in range(n) if q not in keep for part in "xz"])


def positive_css(gens, n: int) -> bool:
    """<gens> is generated by + pure-X and + pure-Z elements."""
    xs = _eliminate(gens, [(q, "z") for q in range(n)])
    zs = _eliminate(gens, [(q, "x") for q in range(n)])
    return all(r.sign == 1 for r in xs + zs) and _rank(xs) + _rank(zs) == _rank(gens)


def decomposes(gens, blocks, n: int) -> bool:
    """No element of <gens> spans two blocks of qubits."""
    return sum(_rank(restrict(gens, set(b), n)) for b in blocks) == _rank(gens)


def to_local(gens, frames: dict, n: int):
    """gens, supported on the qubits of `frames` ({qubit: frame}), conjugated into those local frames."""
    out = []
    for g in gens:
        sign, text = int(g.sign.real), ["_"] * n
        for q in range(n):
            if g[q]:
                s, text[q] = LOCAL[frames[q]]["_XYZ"[g[q]]]
                sign *= s
        out.append(stim.PauliString(("+" if sign == 1 else "-") + "".join(text)))
    return out


# ------------------------------------------------------------------ assumption (i)
def scope_violations(circuit: stim.Circuit) -> list[str]:
    """Operations outside assumption (i)."""
    bad = []
    for k, inst in enumerate(circuit.flattened()):
        name, gd = inst.name, stim.gate_data(inst.name)
        if name in ANNOTATIONS:
            continue
        if any(t.is_measurement_record_target or t.is_sweep_bit_target for t in inst.targets_copy()):
            bad.append(f"{k}: classically controlled {name}")
        elif not (name == "MPAD" or name in RESET_BASIS or name in MEAS_BASIS or name in PAULI_NOISE
                  or (gd.is_unitary and (gd.is_single_qubit_gate or gd.is_two_qubit_gate))):
            bad.append(f"{k}: {name}")
    return bad


# ------------------------------------------------------------------ the frame problem of a circuit and loss model
@dataclass
class FrameProblem:
    """One frame variable per segment (the frame at the segment's start).  domains[s]: bitmask of the frames the
    unary rules allow; into[y]: (x, support) pairs, support[f] the bitmask of y's frames compatible with x in
    frame f; prefix: (qubit, segment) of the used post-cut segments; readouts: (record, segment, gates so far,
    basis, inverted) after the prefix; links: (segment, segment, gate) of the two-qubit gates after the prefix."""
    n_qubits: int
    prefix_end: int
    domains: list
    into: list
    prefix: list
    prefix_state: list
    readouts: list
    reference: np.ndarray
    post_cut: list
    links: list
    conflict: str | None = None
    reasons: list = field(default_factory=list)


def is_reload(inst) -> bool:
    return inst.name == "I_ERROR" and inst.tag == "reload"


def _reference_circuit(flat: stim.Circuit) -> stim.Circuit:
    """The Pauli-free noiseless circuit the masked frame takes its reference from."""
    out = stim.Circuit()
    for inst in flat:
        if inst.name not in INERT:
            out.append(inst.name, inst.targets_copy(), [] if inst.name in MEAS_BASIS else inst.gate_args_copy())
    return out


class _Discarding:
    """A TableauSimulator whose resets discard exactly: the old state moves into a fresh ancilla that nothing
    touches again (a partial trace), instead of Stim's reset, which measures and so collapses entangled partners
    onto a random branch."""

    def __init__(self, n: int):
        self.sim = stim.TableauSimulator()
        self.sim.set_num_qubits(n)
        self.spare = itertools.count(n)

    def prepare(self, q: int, state: str):
        """Discard q and put it in the + eigenstate of `state` ("X", "Y", "Z") or maximally mixed ("mixed")."""
        sim = self.sim
        sim.swap(q, next(self.spare))
        if state in "XY":
            sim.h(q)
        if state == "Y":
            sim.s(q)
        if state == "mixed":
            a = next(self.spare)
            sim.h(a)
            sim.cnot(a, q)

    def postselect(self, basis: str, q: int, value: bool) -> bool:
        try:
            {"X": self.sim.postselect_x, "Y": self.sim.postselect_y, "Z": self.sim.postselect_z}[basis](
                q, desired_value=value)
            return True
        except ValueError:
            return False

    def stabilizers(self, n: int):
        """Generators of the (possibly mixed) state of qubits 0..n-1."""
        sim = self.sim
        out = []
        for g in restrict(sim.canonical_stabilizers(), range(n), sim.num_qubits):
            t = g[:n]
            t.sign = g.sign
            out.append(t)
        return out


def _prefix_state(flat: stim.Circuit, k0: int, reference: np.ndarray) -> list:
    """Stabilizer generators of the state after instructions 0..k0, on the branch that reads the reference."""
    n = flat.num_qubits
    sim = _Discarding(n)
    m = 0
    for inst in flat[: k0 + 1]:
        if inst.name == "MPAD":
            m += len(inst.targets_copy())
        elif inst.name in MEAS_BASIS:
            basis = MEAS_BASIS[inst.name]
            for t in inst.targets_copy():
                assert sim.postselect(basis, t.value, bool(reference[m]) ^ t.is_inverted_result_target)
                m += 1
                if inst.name.startswith("MR"):
                    sim.prepare(t.value, basis)
        elif inst.name in RESET_BASIS:
            for t in inst.targets_copy():
                sim.prepare(t.value, RESET_BASIS[inst.name])
        elif inst.name not in INERT:
            sim.sim.do(inst)
    return sim.stabilizers(n)


def _may_be_lost(flat, model: LossModel, locs):
    """lost(q, k): q may be missing just before instruction k, i.e. a loss location before k with no certain
    refill since (a reload marker with p = 1, or with reload="reset" a reset or reset-readout)."""
    refill, by_qubit = {}, {}
    for k, inst in enumerate(flat):
        certain = is_reload(inst) and (not inst.gate_args_copy() or inst.gate_args_copy()[0] >= 1)
        at_reset = model.reload == "reset" and (inst.name in RESET_BASIS or inst.name.startswith("MR"))
        if certain or at_reset:
            for t in inst.targets_copy():
                refill.setdefault(t.value, []).append(k)
    for k, q in locs:
        by_qubit.setdefault(q, []).append(k)

    def lost(q: int, k: int) -> bool:
        last = max((r for r in refill.get(q, ()) if r < k), default=-2)
        return any(last <= kk < k for kk in by_qubit.get(q, ()))

    return lost


def _shift(row: int, a: int) -> int:
    """`row` is a bitmask over the frames g' = a g; return it as a bitmask over g."""
    return sum(1 << g for g in range(24) if (row >> MUL[a][g]) & 1)


def frame_problem(circuit: stim.Circuit, model: LossModel, reload_points: str = "literal") -> FrameProblem:
    """The frame-CSS rules of `circuit` under `model` as a constraint problem over the segments' frames.
    reload_points="literal" checks rule 4 at every reload point where the qubit may be missing: reload markers,
    and with reload="anywhere" every instruction boundary after the prefix, the end of the circuit included (and its
    start under from_start); "live" only at those followed by a readout or two-qubit gate of the same segment."""
    assert reload_points in ("literal", "live"), reload_points
    flat = circuit.flattened()
    n, N = flat.num_qubits, len(flat)
    locs = loss_locations(flat, model)
    k0 = min((k for k, _ in locs), default=N - 1)
    lost = _may_be_lost(flat, model, locs)
    reference = np.array(_reference_circuit(flat).reference_sample(), dtype=bool)

    domains, last_use, binary, readouts, reloads, reasons = [], [], [], [], [], []
    cur, acc = [0] * n, [ID] * n
    clock = itertools.count()
    p = FrameProblem(n, k0, domains, [], [], [], readouts, reference, [], [])

    def new_segment(q: int) -> int:
        domains.append(ALL)
        last_use.append(-1)
        cur[q], acc[q] = len(domains) - 1, ID
        return cur[q]

    def require(seg: int, a: int, pred, why: str):
        domains[seg] &= mask(lambda f: pred(MUL[a][f]))
        if not domains[seg] and p.conflict is None:
            p.conflict = why

    def reload_point(q: int, k: int):
        if model.reload_state == "zero" and lost(q, k):
            reloads.append((cur[q], acc[q], next(clock), f"reload of qubit {q} before instruction {k}"))

    p.post_cut = [new_segment(q) for q in range(n)]
    m = flat[: k0 + 1].num_measurements
    for k in range(k0 + 1, N):
        inst = flat[k]
        name, targets = inst.name, inst.targets_copy()
        if model.reload == "anywhere":
            for q in range(n):
                reload_point(q, k)
        if is_reload(inst):
            for t in targets:
                reload_point(t.value, k)
        elif name == "MPAD":
            m += len(targets)
        elif name in INERT:
            continue
        elif name in GATE:
            for t in targets:
                acc[t.value] = MUL[GATE[name]][acc[t.value]]
        elif name in RESET_BASIS:
            for t in targets:
                s = new_segment(t.value)
                require(s, ID, lambda f, b=RESET_BASIS[name]: prep_ok(f, b), f"reset of qubit {t.value} at {k}")
        elif name in MEAS_BASIS:
            basis = MEAS_BASIS[name]
            for t in targets:
                q, old, a = t.value, cur[t.value], acc[t.value]
                require(old, a, lambda f, b=basis: reads_channel(f, b), f"readout of qubit {q} at {k}")
                readouts.append((m, old, a, basis, t.is_inverted_result_target))
                last_use[old] = next(clock)
                m += 1
                s = new_segment(q)
                if name.startswith("MR"):
                    require(s, ID, lambda f, b=basis: prep_ok(f, b), f"reset of qubit {q} after its readout at {k}")
                else:
                    sup = [mask(lambda g, fa=MUL[a][f], b=basis: reads_channel(fa, b) and reads_channel(g, b)
                                and LOCAL[g][b][0] == LOCAL[fa][b][0]) for f in range(24)]
                    binary.append((old, s, sup, f"sign kept by qubit {q} after its readout at {k}"))
        else:
            gp = good_pairs(name)
            qs = [t.value for t in targets]
            for a, b in zip(qs[::2], qs[1::2]):
                sa, sb, xa, xb = cur[a], cur[b], acc[a], acc[b]
                binary.append((sa, sb, [_shift(gp[MUL[xa][f]], xb) for f in range(24)], f"{name} {a} {b} at {k}"))
                p.links.append((sa, sb, name))
                last_use[sa] = last_use[sb] = next(clock)
    if model.reload == "anywhere":
        for q in range(n):
            reload_point(q, N)

    for seg, a, when, why in reloads:
        if reload_points == "literal" or last_use[seg] > when:
            require(seg, a, lambda f: prep_ok(f, "Z"), why)
    p.into = [[] for _ in domains]
    for x, y, sup, why in binary:
        p.into[y].append((x, sup))
        p.into[x].append((y, [mask(lambda f, g=g, s=sup: (s[f] >> g) & 1) for g in range(24)]))
        reasons.append(why)
    p.reasons = reasons
    p.prefix = [(q, p.post_cut[q]) for q in range(n) if last_use[p.post_cut[q]] >= 0]
    p.prefix_state = _prefix_state(flat, k0, reference)
    return p


# ------------------------------------------------------------------ O1: backtracking search
def _bits(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def _propagate(dom, into, changed) -> bool:
    """Arc consistency from the variables in `changed`; False when a domain empties."""
    queue = list(changed)
    while queue:
        y = queue.pop()
        for x, sup in into[y]:
            keep = sum(1 << f for f in _bits(dom[x]) if sup[f] & dom[y])
            if keep != dom[x]:
                if not keep:
                    return False
                dom[x] = keep
                queue.append(x)
    return True


def solve(p: FrameProblem):
    """A frame per segment satisfying every rule, or None."""
    dom = list(p.domains)
    if not all(dom) or not _propagate(dom, p.into, range(len(dom))):
        return None
    n = p.n_qubits
    groups = [restrict(p.prefix_state, {q for q, _ in p.prefix[: i + 1]}, n) for i in range(len(p.prefix))]

    def search(dom, i):
        if i < len(p.prefix):
            seg = p.prefix[i][1]
            for f in _bits(dom[seg]):
                d = dom.copy()
                d[seg] = 1 << f
                frames = {q: next(_bits(d[s])) for q, s in p.prefix[: i + 1]}
                if (positive_css(to_local(groups[i], frames, n), n) and _propagate(d, p.into, [seg])
                        and (r := search(d, i + 1)) is not None):
                    return r
            return None
        free = [s for s in range(len(dom)) if dom[s] & (dom[s] - 1) and p.into[s]]
        if not free:
            return [next(_bits(x)) for x in dom]
        seg = min(free, key=lambda s: dom[s].bit_count())
        for f in _bits(dom[seg]):
            d = dom.copy()
            d[seg] = 1 << f
            if _propagate(d, p.into, [seg]) and (r := search(d, i)) is not None:
                return r
        return None

    return search(dom, 0)


def kappa_of(p: FrameProblem, frames) -> np.ndarray:
    """The reference: readout signs in the frames after the prefix, the loss-free reference before it."""
    kappa = p.reference.copy()
    for m, seg, a, basis, inverted in p.readouts:
        kappa[m] = (LOCAL[MUL[a][frames[seg]]][basis][0] == -1) ^ inverted
    return kappa


def brute_force_frame_css(circuit: stim.Circuit, model: LossModel, reload_points: str = "literal"):
    """O1: (True, kappa) when some frame per segment satisfies the frame-CSS rules, else (False, None)."""
    p = frame_problem(circuit, model, reload_points)
    frames = solve(p)
    return (False, None) if frames is None else (True, kappa_of(p, frames))


# ------------------------------------------------------------------ assumption (iii)
def prefix_spans(circuit: stim.Circuit, model: LossModel) -> tuple[bool, bool]:
    """(some stabilizer of the prefix output spans two role components, some stabilizer of its reduction to the
    used qubits does).  Role components: segments joined by the two-qubit gates after the prefix that are CSS in
    some frames (II, CSS in every frame, ties no roles)."""
    p = frame_problem(circuit, model)
    parent = list(range(len(p.domains)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, name in p.links:
        if name in CSS_ABLE_2Q:
            parent[find(a)] = find(b)
    n = p.n_qubits

    def blocks(qubits):
        out = {}
        for q in qubits:
            out.setdefault(find(p.post_cut[q]), []).append(q)
        return list(out.values())

    used = [q for q, _ in p.prefix]
    full = not decomposes(p.prefix_state, blocks(range(n)), n)
    reduced = not decomposes(restrict(p.prefix_state, set(used), n), blocks(used), n)
    return full, reduced


# ------------------------------------------------------------------ O2: common reference
def loss_patterns(circuit: stim.Circuit, model: LossModel, cap: int = 1 << 12, n_random: int = 100,
                  n_pairs: int | None = None, seed: int = 0):
    """Every subset of the loss locations when there are at most `cap`; else no loss, every single loss, the
    pairs of losses (all, or `n_pairs` random ones) and `loss_family`'s random multi-loss patterns."""
    flat = circuit.flattened()
    locs = loss_locations(flat, model)

    def pattern(slots):
        out = {}
        for k, q in slots:
            out.setdefault(k, []).append(q)
        return out

    if 1 << len(locs) <= cap:
        return [pattern(c) for r in range(len(locs) + 1) for c in itertools.combinations(locs, r)]
    pairs = list(itertools.combinations(locs, 2))
    if n_pairs is not None and n_pairs < len(pairs):
        rng = np.random.default_rng(seed)
        pairs = [pairs[int(i)] for i in rng.choice(len(pairs), n_pairs, replace=False)]
    fam = loss_family(flat, reload=model.reload == "reset", slots=locs, n_random=n_random, seed=seed) if n_random else []
    return [{}] + [pattern([s]) for s in locs] + [pattern(c) for c in pairs] + fam


def common_reference_failures(circuit: stim.Circuit, model: LossModel, kappa, patterns) -> list:
    """O2: the patterns s for which kappa is not a possible noiseless record of the lossy circuit C_s.  Exact for
    reload_state="zero" with reload "marked" or "reset"; with reload="anywhere" the patterns reload only at
    markers, a subset of the model's patterns.  Not valid for reload_state="mixed" (C_s reloads |0>)."""
    assert model.reload_state == "zero", "constraints() reloads |0>"
    r = sum(1 << int(i) for i in np.flatnonzero(kappa))
    reload = model.reload == "reset"
    return [s for s in patterns if any(((y & r).bit_count() & 1) != v for y, v in constraints(circuit, s, reload))]



def reads_reference(circuit: stim.Circuit, model: LossModel, kappa, losses: dict, reloads: dict | None = None) -> bool:
    """kappa is a possible noiseless record of the lossy circuit, simulated directly (independent of
    `constraints`): losses {k: qubits lost right after instruction k, -1 before the start}; reloads {k: qubits
    refilled just before instruction k (k = len: at the end)}, only meaningful with reload="anywhere".  A refill
    (a reload marker, a reset with reload="reset", or a boundary reload) puts |0>, or with reload_state="mixed" a
    maximally mixed atom, into an empty slot; a lost readout is a free coin."""
    flat = circuit.flattened()
    sim = _Discarding(flat.num_qubits)
    lost = set(losses.get(-1, ()))
    reloads = reloads or {}

    def refill(q: int):
        if q in lost:
            lost.discard(q)
            sim.prepare(q, "Z" if model.reload_state == "zero" else "mixed")

    m = 0
    for k, inst in enumerate(flat):
        for q in reloads.get(k, ()):
            refill(q)
        name, targets = inst.name, inst.targets_copy()
        if is_reload(inst):
            for t in targets:
                refill(t.value)
        elif name == "MPAD":
            for t in targets:
                if bool(kappa[m]) != bool(t.value):
                    return False
                m += 1
        elif name in MEAS_BASIS or name in RESET_BASIS:
            basis = MEAS_BASIS.get(name) or RESET_BASIS[name]
            for t in targets:
                q = t.value
                if name in MEAS_BASIS:
                    if q not in lost and not sim.postselect(basis, q, bool(kappa[m]) ^ t.is_inverted_result_target):
                        return False
                    m += 1
                if name in RESET_BASIS or name.startswith("MR"):
                    if model.reload == "reset":
                        refill(q)
                    if q not in lost:
                        sim.prepare(q, basis)
        elif name not in INERT:
            w = 2 if name in TWO_Q else 1
            qs = [t.value for t in targets]
            for i in range(0, len(qs), w):
                if not set(qs[i:i + w]) & lost:
                    sim.sim.do(stim.CircuitInstruction(name, qs[i:i + w]))
        lost |= set(losses.get(k, ()))
    return True


def random_events(circuit: stim.Circuit, model: LossModel, rng, max_losses: int = 3):
    """A random loss pattern of the model (a few of its loss locations), plus random boundary reloads when
    reload="anywhere"."""
    flat = circuit.flattened()
    locs = loss_locations(flat, model)
    losses, reloads = {}, {}
    for i in rng.permutation(len(locs))[: int(rng.integers(0, max_losses + 1))]:
        k, q = locs[int(i)]
        losses.setdefault(k, []).append(q)
    if model.reload == "anywhere" and flat.num_qubits:
        for _ in range(int(rng.integers(0, max_losses + 1))):
            reloads.setdefault(int(rng.integers(0, len(flat) + 1)), []).append(int(rng.integers(flat.num_qubits)))
    return losses, reloads


def simulated_failures(circuit: stim.Circuit, model: LossModel, kappa, rng, n_patterns: int = 50) -> list:
    """O2 by direct simulation for every loss model, reload="anywhere" and reload_state="mixed" included: the
    random event patterns (losses, reloads) under which kappa is not a possible noiseless record."""
    out = []
    for _ in range(n_patterns):
        losses, reloads = random_events(circuit, model, rng)
        if not reads_reference(circuit, model, kappa, losses, reloads):
            out.append((losses, reloads))
    return out

# ------------------------------------------------------------------ O3: circuits
DEFAULT_MODEL = LossModel()
MOVERS = tuple(g for g in ONE_Q if g not in PAULI_GATES and g != "I")   # Pauli gates leave the analysed frame alone


@dataclass(frozen=True)
class GateSet:
    two: tuple   # two-qubit gates
    one: tuple   # single-qubit gates that move frames and appear as random gates


GATE_SETS = {
    "css": GateSet(("CX", "SWAP", "CXSWAP"), MOVERS),
    "cz_h": GateSet(("CZ",), ("H",)),
    "cy_s": GateSet(("CY",), ("S",)),
    "h_s": GateSet(("CX", "CZ", "SWAP"), ("H", "S")),
    "clifford": GateSet(CSS_ABLE_2Q, MOVERS),
}


class Emitter:
    """A circuit under construction whose instructions never fuse (Stim merges adjacent identical ones)."""

    def __init__(self):
        self.circuit = stim.Circuit()

    def __call__(self, name: str, targets, args=(), tag: str = ""):
        c = self.circuit
        if len(c) and c[-1].name == name and c[-1].tag == tag and list(c[-1].gate_args_copy()) == list(args):
            c.append("TICK")
        c.append(name, targets, list(args), tag=tag)


def joined(*parts) -> stim.Circuit:
    """Concatenation with a TICK between parts, so that no instruction fuses across a junction."""
    out = stim.Circuit()
    for part in parts:
        if len(out) and len(part):
            out.append("TICK")
        out += part
    return out


def _words_from(src: int, gens, ok) -> dict:
    """{frame: shortest gate word reaching it from src}, every frame on the way (target included) ok."""
    words, frontier = {src: []}, [src]
    while frontier:
        nxt = []
        for f in frontier:
            for g in gens:
                h = MUL[GATE[g]][f]
                if h not in words and ok(h):
                    words[h] = words[f] + [g]
                    nxt.append(h)
        frontier = nxt
    return words


def _gf2_nullspace(rows, k: int):
    """Basis of {b : r.b = 0 for every row r}, vectors as k-bit ints."""
    piv = {}
    for r in rows:
        for h, p in piv.items():
            if (r >> h) & 1:
                r ^= p
        if r:
            h = r.bit_length() - 1
            for hh in list(piv):
                if (piv[hh] >> h) & 1:
                    piv[hh] ^= r
            piv[h] = r
    free = [i for i in range(k) if i not in piv]
    return [(1 << j) | sum(1 << h for h, p in piv.items() if (p >> j) & 1) for j in free]


def css_state_circuit(rng, qubits, frames, ghz: bool = False) -> stim.Circuit:
    """A `noiseless` circuit from |0..0> to a positive CSS state on `qubits` hidden behind `frames`
    (local -> physical: the physical state is (x) F_q |psi_local>).  ghz: the GHZ state, entangled across
    every cut; else a random CSS state."""
    k = len(qubits)
    xs = [(1 << k) - 1] if ghz else [int(rng.integers(1, 1 << k)) for _ in range(int(rng.integers(0, k + 1)))]
    zs = _gf2_nullspace(xs, k)
    xs = _gf2_nullspace(zs, k)
    stabs = []
    for vec, axis in [(v, "X") for v in xs] + [(v, "Z") for v in zs]:
        sign, text = 1, []
        for i in range(k):
            if (vec >> i) & 1:
                image = TABLEAU[frames[i]](stim.PauliString(axis))
                sign *= int(image.sign.real)
                text.append("_XYZ"[image[0]])
            else:
                text.append("_")
        stabs.append(stim.PauliString(("+" if sign == 1 else "-") + "".join(text)))
    out = stim.Circuit()
    for inst in stim.Tableau.from_stabilizers(stabs).to_circuit("elimination"):
        out.append(inst.name, [qubits[t.value] for t in inst.targets_copy()], inst.gate_args_copy(), tag="noiseless")
    return out


def _is_loss_location(model: LossModel, name: str, tag: str) -> bool:
    if tag == "noiseless" and model.noiseless_loss_free:
        return False
    if model.after == "2q":
        return name in TWO_Q
    return name not in PAULI_NOISE and name not in ANNOTATIONS and name != "MPAD"


def random_frame_css(rng, n_qubits: int, depth: int, gate_set: str = "css", model: LossModel = DEFAULT_MODEL,
                     prefix: str = "resets", extras: bool = True) -> stim.Circuit:
    """O3: a circuit that is frame-CSS under `model` by construction.  The generator keeps one frame per qubit
    and, before every operation, inserts the single-qubit gates that put the frames where the rules hold.
    prefix: "resets" (each qubit starts with a reset), "encoder" (a noiseless encoder of a positive CSS state
    hidden behind random frames, on qubits that later two-qubit gates link) or "split" (a noiseless GHZ state
    whose halves no later two-qubit gate links: assumption (iii) fails, frame-CSS is not guaranteed).
    extras: Pauli gates, noise, annotations, MPAD and `noiseless` tags, none of which changes the frames."""
    gs = GATE_SETS[gate_set]
    n = n_qubits
    e = Emitter()
    zero = model.reload_state == "zero"
    ok = (lambda f: prep_ok(f, "Z")) if zero and model.reload == "anywhere" else (lambda f: True)
    frame = [ID] * n
    started = model.from_start
    side = [0] * n
    order = [int(q) for q in rng.permutation(n)]

    def emit(name, targets, args=(), tag=""):
        nonlocal started
        e(name, targets, args, tag)
        started = started or _is_loss_location(model, name, tag)

    def pick(options):
        return options[int(rng.integers(len(options)))] if options else None

    def move(q: int, word):
        for g in word:
            emit(g, [q])
            frame[q] = MUL[GATE[g]][frame[q]]

    def fresh(pred) -> int:
        return pick([f for f in range(24) if pred(f) and ok(f)])

    def reset(q: int):
        name = pick(list(RESET_BASIS))
        emit(name, [q])
        frame[q] = fresh(lambda f: prep_ok(f, RESET_BASIS[name]))

    def readout(q: int, name: str) -> bool:
        basis = MEAS_BASIS[name]
        words = _words_from(frame[q], gs.one, ok)
        target = pick([f for f in words if reads_channel(f, basis)])
        if target is None:
            return False
        move(q, words[target])
        sign = LOCAL[target][basis][0]
        emit(name, [stim.target_inv(q) if rng.random() < 0.3 else q])
        if name.startswith("MR"):
            frame[q] = fresh(lambda f: prep_ok(f, basis))
        else:
            frame[q] = fresh(lambda f: reads_channel(f, basis) and LOCAL[f][basis][0] == sign)
        return True

    def two_qubit(a: int, b: int, name: str) -> bool:
        wa, wb = _words_from(frame[a], gs.one, ok), _words_from(frame[b], gs.one, ok)
        gp = good_pairs(name)
        pair = pick([(fa, fb) for fa in wa for fb in _bits(gp[fa]) if fb in wb])
        if pair is None:
            return False
        move(a, wa[pair[0]])
        move(b, wb[pair[1]])
        emit(name, [a, b])
        return True

    def extra(q: int):
        kind = int(rng.integers(6))
        if kind == 0:
            emit(pick(sorted(PAULI_GATES)), [q])
        elif kind == 1:
            emit(pick(["X_ERROR", "Z_ERROR", "DEPOLARIZE1"]), [q], [0.01])
        elif kind == 2:
            emit("TICK", [])
        elif kind == 3 and e.circuit.num_measurements:
            emit("DETECTOR", [stim.target_rec(-1)])
        elif kind == 4:
            emit("MPAD", [int(rng.integers(2))])
        elif kind == 5:
            g = pick([g for g in gs.one if ok(MUL[GATE[g]][frame[q]])])
            if g is not None:
                emit(g, [q], tag="noiseless")
                frame[q] = MUL[GATE[g]][frame[q]]

    linked = []
    if prefix == "resets":
        for q in order:
            reset(q)
    else:
        assert model.noiseless_loss_free and not model.from_start, "the encoder must be loss-free"
        size = int(rng.integers(2, n + 1))
        hidden = sorted(order[:size])
        reachable = list(_words_from(ID, gs.one, ok))
        for q in hidden:
            frame[q] = pick(reachable)
        if prefix == "split":
            cut = int(rng.integers(1, size))
            side = [int(q in hidden[cut:] or (q not in hidden and rng.random() < 0.5)) for q in range(n)]
        e.circuit += css_state_circuit(rng, hidden, [frame[q] for q in hidden], ghz=prefix == "split")
        for q in order:
            if q not in hidden:
                reset(q)
        if prefix == "split":
            emit("I", [hidden[0]])
        else:
            linked = [(hidden[i], hidden[int(rng.integers(i))]) for i in range(1, size)]
            linked.append(linked[0])
    for a, b in linked:
        assert any(two_qubit(a, b, pick(list(gs.two))) for _ in range(50)), f"no gate of {gate_set} links {a}, {b}"

    for _ in range(depth):
        kind = pick(["2q"] * 4 + ["1q"] * 2 + ["readout"] * 2 + ["reset", "reload"] + (["extra"] if extras else []))
        q = int(rng.integers(n))
        if kind == "2q":
            partners = [b for b in range(n) if b != q and side[b] == side[q]]
            if partners:
                two_qubit(q, pick(partners), pick(list(gs.two)))
        elif kind == "1q":
            g = pick([g for g in gs.one if ok(MUL[GATE[g]][frame[q]])])
            if g is not None:
                emit(g, [q])
                frame[q] = MUL[GATE[g]][frame[q]]
        elif kind == "readout" and started:
            readout(q, pick(list(MEAS_BASIS)))
        elif kind == "reset" and started:
            reset(q)
        elif kind == "reload":
            words = _words_from(frame[q], gs.one, ok)
            target = pick([f for f in words if prep_ok(f, "Z")])
            if target is not None:
                move(q, words[target])
                emit("I_ERROR", [q], [0.5] if rng.random() < 0.5 else [], tag="reload")
        elif kind == "extra":
            extra(q)
    if not started:
        emit("I", [0])
    for q in order:
        while not readout(q, pick(["M", "MX", "MY"])):
            pass
    return e.circuit


def random_clifford(rng, n_qubits: int, n_ops: int, two=TWO_Q, markers: bool = True) -> stim.Circuit:
    """A random circuit inside assumption (i): resets, single- and two-qubit Cliffords from `two`, readouts with
    inverted targets, mid-circuit resets, reload markers, MPAD and `noiseless` tags."""
    n = n_qubits
    e = Emitter()
    two = [g for g in two if g != "II"]
    for q in range(n):
        e(str(rng.choice(list(RESET_BASIS))), [q])
    for _ in range(n_ops):
        kind = int(rng.integers(10))
        q = int(rng.integers(n))
        tag = "noiseless" if rng.random() < 0.1 else ""
        if kind < 4 and n > 1:
            b = int(rng.choice([b for b in range(n) if b != q]))
            e(str(rng.choice(two)), [q, b], tag=tag)
        elif kind < 7:
            e(str(rng.choice(ONE_Q)), [q], tag=tag)
        elif kind == 7:
            e(str(rng.choice(list(MEAS_BASIS))), [stim.target_inv(q) if rng.random() < 0.3 else q], tag=tag)
        elif kind == 8:
            e(str(rng.choice(list(RESET_BASIS))), [q], tag=tag)
        elif markers:
            e("I_ERROR", [q], [0.5] if rng.random() < 0.5 else [], tag="reload")
    for q in range(n):
        e(str(rng.choice(["M", "MX", "MY"])), [q])
    return e.circuit


def _prefix_end(circuit: stim.Circuit, model: LossModel) -> int:
    return min((k for k, _ in loss_locations(circuit, model)), default=len(circuit) - 1)


def _insert(flat: stim.Circuit, p: int, text: str) -> stim.Circuit:
    return joined(flat[:p], stim.Circuit(text), flat[p:])


def mutations(circuit: stim.Circuit, rng, model: LossModel):
    """Single-edit near misses after the prefix: {kind: circuit}.  Kinds: flip (swap a two-qubit gate's
    operands), stray (a lone single-qubit gate), basis (another readout basis), triangle (CZ on three qubits),
    three_axes (a qubit asked to carry X, Y and Z), loop (a permutation cycle closing with an H), sign (S S or
    SQRT_X SQRT_X: a Pauli made of non-Pauli gates)."""
    flat = circuit.flattened()
    n, N = flat.num_qubits, len(flat)
    k0 = _prefix_end(flat, model)
    p = int(rng.integers(k0 + 1, N + 1)) if k0 + 1 <= N else N
    qs = [int(q) for q in rng.permutation(max(n, 3))[:3]]
    a, b, c = qs
    out = {
        "stray": _insert(flat, p, f"{rng.choice(MOVERS)} {a}"),
        "triangle": _insert(flat, p, f"CZ {a} {b}\nTICK\nCZ {b} {c}\nTICK\nCZ {a} {c}"),
        "three_axes": _insert(flat, p, f"CX {a} {b}\nTICK\nCX {b} {a}\nTICK\nCY {c} {a}"),
        "loop": _insert(flat, p, f"SWAP {a} {b}\nTICK\nH {a}\nTICK\nSWAP {a} {b}"),
        "sign": _insert(flat, p, f"{rng.choice(['S', 'SQRT_X', 'SQRT_Y'])} {a} {a}"),
    }
    later = [k for k in range(k0 + 1, N)]
    two = [k for k in later if flat[k].name in TWO_Q and flat[k].name != "II"]
    if two:
        k = int(rng.choice(two))
        ts = flat[k].targets_copy()
        i = 2 * int(rng.integers(len(ts) // 2))
        ts[i], ts[i + 1] = ts[i + 1], ts[i]
        out["flip"] = joined(flat[:k], _instruction(flat[k], ts), flat[k + 1:])
    reads = [k for k in later if flat[k].name in MEAS_BASIS]
    if reads:
        k = int(rng.choice(reads))
        name = flat[k].name
        other = str(rng.choice([m for m in MEAS_BASIS if m != name and m.startswith("MR") == name.startswith("MR")]))
        out["basis"] = joined(flat[:k], _instruction(flat[k], flat[k].targets_copy(), other), flat[k + 1:])
    return out


def _instruction(inst, targets, name=None) -> stim.Circuit:
    out = stim.Circuit()
    out.append(name or inst.name, targets, inst.gate_args_copy(), tag=inst.tag)
    return out


def _map_target(t, perm):
    if t.is_qubit_target:
        return stim.target_inv(perm[t.value]) if t.is_inverted_result_target else perm[t.value]
    if t.is_x_target or t.is_y_target or t.is_z_target:
        make = stim.target_x if t.is_x_target else stim.target_y if t.is_y_target else stim.target_z
        return make(perm[t.value], t.is_inverted_result_target)
    return t


def rewrites(circuit: stim.Circuit, rng, model: LossModel):
    """Rewrites that keep the frame-CSS verdict under `model`: {kind: circuit}.  relabel (permute qubits),
    pair (H H in one instruction, S then S_DAG, or X then X, right after an operation on the qubit; with reloads
    anywhere only pairs that commute with Z, since a reload may land between the two gates), split
    (an instruction on pairwise-disjoint qubits into two), fuse (two such instructions separated by a TICK into
    one), inert (a TICK, DETECTOR or Pauli noise channel).  Edits stay after the prefix end, so the prefix, the
    loss locations up to ghost equivalence and the frames at the reload points are kept."""
    flat = circuit.flattened()
    n, N = flat.num_qubits, len(flat)
    k0 = _prefix_end(flat, model)
    out = {}
    perm = [int(q) for q in rng.permutation(n + 2)[:n]]
    relabeled = stim.Circuit()
    for inst in flat:
        ts = inst.targets_copy() if inst.name == "MPAD" else [_map_target(t, perm) for t in inst.targets_copy()]
        relabeled.append(inst.name, ts, inst.gate_args_copy(), tag=inst.tag)
    out["relabel"] = relabeled

    ops = [k for k in range(k0 + 1, N) if flat[k].name not in INERT and flat[k].name != "MPAD"
           and not is_reload(flat[k]) and flat[k].tag != "noiseless"]
    if ops:
        k = int(rng.choice(ops))
        q = int(rng.choice([t.value for t in flat[k].targets_copy()]))
        pairs = [f"H {q} {q}", f"S {q}\nTICK\nS_DAG {q}", f"X {q}\nTICK\nX {q}"]
        if model.reload == "anywhere" and model.reload_state == "zero":
            pairs = [f"S {q}\nTICK\nS_DAG {q}", f"Z {q} {q}"]    # a reload may land between the two gates
        pair = str(rng.choice(pairs))
        out["pair"] = _insert(flat, k + 1, pair)

    def groups(inst):
        ts = inst.targets_copy()
        w = 2 if stim.gate_data(inst.name).is_two_qubit_gate else 1
        return [ts[i:i + w] for i in range(0, len(ts), w)]

    def disjoint(gs):
        qs = [t.value for g in gs for t in g]
        return len(qs) == len(set(qs))

    splittable = [k for k in range(k0 + 1, N) if flat[k].name not in INERT and flat[k].name != "MPAD"
                  and not is_reload(flat[k]) and len(groups(flat[k])) > 1 and disjoint(groups(flat[k]))]
    if splittable:
        k = int(rng.choice(splittable))
        gs = groups(flat[k])
        cut = int(rng.integers(1, len(gs)))
        first = _instruction(flat[k], [t for g in gs[:cut] for t in g])
        second = _instruction(flat[k], [t for g in gs[cut:] for t in g])
        out["split"] = joined(flat[:k], first, second, flat[k + 1:])
    fusable = [k for k in range(k0 + 1, N - 2) if flat[k + 1].name == "TICK" and flat[k].name == flat[k + 2].name
               and flat[k].name not in INERT and flat[k].name != "MPAD" and flat[k].tag == flat[k + 2].tag
               and flat[k].gate_args_copy() == flat[k + 2].gate_args_copy()
               and disjoint(groups(flat[k]) + groups(flat[k + 2]))]
    if fusable:
        k = int(rng.choice(fusable))
        merged = _instruction(flat[k], flat[k].targets_copy() + flat[k + 2].targets_copy())
        out["fuse"] = joined(flat[:k], merged, flat[k + 3:])
    if N:
        p = int(rng.integers(0, N + 1))
        q = int(rng.integers(n))
        m = flat[:p].num_measurements
        inert = ["TICK", f"X_ERROR(0.1) {q}", f"DEPOLARIZE1(0.01) {q}", f"PAULI_CHANNEL_1(0.01, 0.02, 0.03) {q}",
                 f"E(0.1) X{q}"] + (["DETECTOR rec[-1]"] if m else [])
        out["inert"] = _insert(flat, p, str(rng.choice(inert)))
    return out


def minimize(circuit: stim.Circuit, still_fails) -> stim.Circuit:
    """Greedy shrink: drop whole instructions, then single gates of multi-target instructions, while
    still_fails(circuit) holds (an exception counts as not failing)."""

    def fails(c):
        try:
            return still_fails(c)
        except (ValueError, IndexError, AssertionError):
            return False

    cur = circuit.flattened()
    changed = True
    while changed:
        changed = False
        for k in range(len(cur) - 1, -1, -1):
            trial = cur[:k] + cur[k + 1:]
            if fails(trial):
                cur, changed = trial, True
        for k in range(len(cur) - 1, -1, -1):
            inst = cur[k]
            w = 2 if stim.gate_data(inst.name).is_two_qubit_gate else 1
            ts = inst.targets_copy()
            for i in range(len(ts) - w, -1, -w) if len(ts) > w else ():
                trial = joined(cur[:k], _instruction(inst, ts[:i] + ts[i + w:]), cur[k + 1:])
                if fails(trial):
                    cur, changed = trial, True
                    break
            if changed:
                break
    return cur
