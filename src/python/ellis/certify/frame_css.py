"""Frame-CSS: is the circuit CSS in one loss-independent local frame per state segment?

Loss model (`LossModel`).  A loss discards an atom: its later operations are dropped (a two-qubit gate as a
whole) and its readouts become fair coins.  Losses may follow the model's loss locations (`loss_slots`).
A reload puts a fresh atom into an empty slot: at every `I_ERROR[reload](p) q` marker (failing with
probability 1 - p, a loss location right after the marker), plus at resets
(reload="reset") or between any two operations (reload="anywhere").  The fresh atom is |0>, or maximally
mixed (reload_state="mixed").

The proof uses one model only, PESSIMISTIC: any atom may be lost before the first operation and after every
operation on it, and a |0> may be reloaded between any two operations.  Every pattern of every other model is a
PESSIMISTIC pattern or equivalent to one (a maximally mixed reload is a |0> reload followed by Pauli noise; a
reload at a lost MR is a |0> reload just before it), so kappa from a PESSIMISTIC proof is a common reference
under every model.  That model need not call the circuit frame-CSS, since its loss-free prefix fixes readouts
to one branch.  The converse fails: a circuit outside frame-CSS here may still
have a common reference under a weaker model.

Segments.  Each qubit's timeline starts a segment at the start, at resets and at readouts: at each cut its
state is fresh in every loss pattern.  Loss and reload do not cut.  Each segment gets one unknown
single-qubit Clifford frame F, carried through the segment's single-qubit gates (F_t = A_t F, A_t the
product of those gates so far).  The frame names the qubit's two channels: the physical Paulis that read
as X and as Z.  One rule, positive CSS in the frame, is checked for every operation from its tableau:

  prep      a reset, the state a readout leaves, or a reloaded |0> (the initial |0> included) is a +
            eigenstate of one channel;
  gate      a two-qubit gate W^dag G W (W = frames of its operands) maps X channels to + products of X
            channels and Z channels to + products of Z channels;
  readout   reads one channel; its sign is the reference bit kappa_m.

Theorem.  If a frame assignment satisfies the rules, kappa is a common reference for every loss and reload
pattern of PESSIMISTIC, so the plain masked frame with reference kappa is exact.  Proof: follow the record
branch that reads kappa at every random readout.  In the frame the state stays positive CSS: surviving
operations obey the rules, dropped gates do nothing (frames never change across two-qubit gates), a
discard keeps the stabilizers trivial on the lost atom, and a reload adds one + stabilizer.  So every
readout can read its kappa bit (paper, sec. "Frame-CSS circuits").

Scope.  The decision below is exact (it accepts exactly the circuits frame-CSS under PESSIMISTIC) when the
operations are single-qubit Pauli-basis resets and readouts, one- and two-qubit Clifford gates and Pauli noise,
with no classical control (assumption (i)).  A circuit that breaks (i) gets "outside-assumptions".  Roles that
the circuit leaves free (a whole component's X/Z swap, a group's unused second axis) do not affect any rule, so
one choice decides.

Decision procedure (near-linear in the circuit):
  * two-qubit gate shapes, computed once per gate type by brute force over the 6 x 6 frame classes:
    controlled (CX, CY, CZ, XC*, YC*: its two axes carry opposite channels), permutation (SWAP, CXSWAP,
    SWAPCX: one frame determines the other), or never CSS (ISWAP, SQRT_XX, CZSWAP, ...);
  * a weighted union-find joins the segments a permutation gate ties; each group may use two axes;
  * a parity union-find assigns the channel roles;
  * one GF(2) system over the frames' Pauli parts (two bits per group) sets the signs.
"""
import itertools
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
import stim

from ellis import frame as mf
from ellis.certify.constraints import loss_slots

I1 = stim.Tableau(1)
REPS = [stim.Tableau.from_named_gate(g) for g in ("I", "H", "S", "SQRT_X_DAG", "C_XYZ", "C_ZYX")]
PAULI_PART = [stim.Tableau.from_named_gate(g) for g in ("I", "X", "Z", "Y")]   # index x + 2 z
CLASS = {(str(r.x_output(0))[1:], str(r.z_output(0))[1:]): i for i, r in enumerate(REPS)}
OTHER = {"X": "Z", "Z": "X", "Y": "Z"}


# ------------------------------------------------------------------ the loss model
@dataclass(frozen=True)
class LossModel:
    after: str = "2q"                 # loss locations: after each two-qubit gate ("2q") or each operation ("any")
    noiseless_loss_free: bool = True  # operations tagged `noiseless` are not loss locations
    from_start: bool = False          # an atom may also be missing before the first operation
    reload: str = "marked"            # "marked" (I_ERROR[reload] only), "reset" (also resets), "anywhere"
    reload_state: str = "zero"        # a reloaded atom is |0> ("zero") or maximally mixed ("mixed")

    def __post_init__(self):
        assert self.after in ("2q", "any"), self.after
        assert self.reload in ("marked", "reset", "anywhere"), self.reload
        assert self.reload_state in ("zero", "mixed"), self.reload_state

    @property
    def reload_at_reset(self) -> bool:
        """The `reload` flag of frame.sample / explicit_lossy_circuit / constraints."""
        return self.reload == "reset"


PESSIMISTIC = LossModel(after="any", noiseless_loss_free=False, from_start=True, reload="anywhere")


def loss_locations(circuit: stim.Circuit, model: LossModel):
    """(k, q): q may be lost right after instruction k (k = -1: before the first instruction).  A reload marker
    `I_ERROR[reload](p)` with p < 1 adds (k, q) for each target q that may be empty there: its failure."""
    start = [(-1, q) for q in range(circuit.num_qubits)] if model.from_start else []
    locations = start + loss_slots(circuit, model.after, not model.noiseless_loss_free)
    first = {}
    for k, q in locations:
        first[q] = min(first.get(q, k), k)
    for k, inst in enumerate(circuit.flattened()):
        if mf.is_reload(inst) and inst.gate_args_copy() and inst.gate_args_copy()[0] < 1:
            locations += [(k, q) for q in dict.fromkeys(t.value for t in inst.targets_copy()) if first.get(q, k) < k]
    return sorted(locations, key=lambda s: s[0])     # stable: unchanged order without failing reloads


# ------------------------------------------------------------------ frames and gate shapes
def image(U, axis, sign=1):
    """The frame image U^dag (sign * axis) U, as (letter, sign)."""
    p = U.inverse()(stim.PauliString(axis)) * sign
    return "_XYZ"[p[0]], int(p.sign.real)


def channel(c, axis):
    """The channel ('X' or 'Z') a root-coordinate axis carries under frame class c; None for the unused axis."""
    letter = image(REPS[c], axis)[0]
    return None if letter == "Y" else letter


def positive_css(T, signs=True):
    """T maps every X to a product of X's and every Z to a product of Z's (with sign + if signs)."""
    n = len(T)
    for k in range(n):
        for out, bad in ((T.x_output(k), (2, 3)), (T.z_output(k), (1, 2))):
            if any(out[j] in bad for j in range(n)) or (signs and out.sign != 1):
                return False
    return True


def _cls(t):
    return CLASS[(str(t.x_output(0))[1:], str(t.z_output(0))[1:])]


def _shape(name):
    """How a two-qubit gate constrains the frame classes (c_a, c_b) of its operands, by brute force."""
    G = stim.Tableau.from_named_gate(name)
    pairs = list(itertools.product(range(6), repeat=2))
    ok = {(i, j) for i, j in pairs if positive_css((REPS[i] + REPS[j]).inverse() * G * (REPS[i] + REPS[j]), False)}
    if len(ok) == 36:
        return ("free",)
    for P, Q in itertools.product("XYZ", repeat=2):
        if ok == {(i, j) for i, j in pairs if channel(i, P) and channel(j, Q) and channel(i, P) != channel(j, Q)}:
            return ("ctrl", P, Q)
    maps, dom = {_cls(REPS[j] * REPS[i].inverse()) for i, j in ok}, {i for i, _ in ok}
    if ok and len(maps) == 1 and len(dom) == len(ok):
        need = tuple(a for a in "XYZ" if all(channel(i, a) for i in dom))
        if dom == {i for i in range(6) if all(channel(i, a) for a in need)}:
            return ("perm", REPS[maps.pop()], need)
    return ("none",)


SHAPES = {name: _shape(name) for name in sorted(mf.UNITARY_2Q)}
OPERATIONS = mf.UNITARY_1Q | mf.UNITARY_2Q | set(mf.RESET) | set(mf.MEAS)


# ------------------------------------------------------------------ result and checks
Verdict = Literal["frame-css", "not-frame-css", "outside-assumptions"]


@dataclass
class FrameCSS:
    verdict: Verdict
    reference: np.ndarray | None = None          # kappa, for frame.sample(reference=...), when frame-CSS
    segments: int = 0
    groups: int = 0
    reasons: list = field(default_factory=list)  # why not frame-CSS, or which assumption fails

    @property
    def proven(self) -> bool:
        return self.verdict == "frame-css"


class Prep(NamedTuple):      # a + eigenstate of the physical axis, at frame time A of segment seg
    seg: int
    A: stim.Tableau
    axis: str
    sign: int
    k: int
    q: int
    reload: bool


class Link(NamedTuple):      # a readout's post-state: the old frame's sign must equal the new frame's sign
    old: int
    A: stim.Tableau
    new: int
    axis: str
    k: int
    q: int


class Gate(NamedTuple):      # a two-qubit gate in the frames of its operands
    name: str
    sa: int
    Aa: stim.Tableau
    sb: int
    Ab: stim.Tableau
    k: int
    qs: tuple


class _Fail(Exception):
    pass


class _Outside(Exception):
    """The circuit breaks an assumption under which the stages decide frame-CSS exactly."""


def frame_css(circuit: stim.Circuit) -> FrameCSS:
    """Decide frame-CSS under PESSIMISTIC and build kappa, a common reference for every loss model."""
    res = FrameCSS("not-frame-css")
    flat = circuit.flattened()
    try:
        _check_operations(flat)
        _certify(flat, res)
    except _Outside as e:
        res.verdict = "outside-assumptions"
        res.reasons.append(str(e))
    except _Fail as e:
        res.reasons.append(str(e))
    return res


def _frame_free(inst) -> bool:
    """Annotations, noise, Pauli and identity gates: no frame changes and no loss-relevant timing."""
    gd = stim.gate_data(inst.name)
    return (inst.name in mf.ANNOTATIONS or inst.name in mf.PAULI or inst.name in ("I", "II")
            or (gd.is_noisy_gate and not gd.produces_measurements))


def _check_operations(flat):
    """Assumption (i): single-qubit Pauli-basis resets and readouts, one- and two-qubit Clifford gates, Pauli
    noise and annotations, with no classically controlled operation."""
    for k, inst in enumerate(flat):
        gd = stim.gate_data(inst.name)
        if inst.name in mf.ANNOTATIONS or (gd.is_noisy_gate and not gd.produces_measurements):
            continue
        if inst.name not in OPERATIONS:
            raise _Outside(f"assumption (i): instruction {k} `{inst}` is not a single-qubit reset or readout, "
                           f"a one- or two-qubit Clifford gate, or Pauli noise")
        if any(t.is_measurement_record_target or t.is_sweep_bit_target for t in inst.targets_copy()):
            raise _Outside(f"assumption (i): instruction {k} `{inst}` is classically controlled")


def _certify(flat, res):
    n, ops = flat.num_qubits, list(flat)

    # ---------------------------------------------------------------- 1. timeline: segments and what they must satisfy
    segs, cur, A = [], {}, {}
    uses = []        # (segment, A, physical axis, why): this axis must carry a channel
    perms = []       # (seg_a, seg_b, M, why): frame of seg_b = M frame of seg_a
    ctrls = []       # ((seg_a, A_a, P), (seg_b, A_b, Q), why): P and Q carry opposite channels
    checks = []      # Prep / Link / Gate, signed once the frames are fixed
    meas = []        # (record index, segment, A, axis, inverted)
    kappa = np.zeros(flat.num_measurements, bool)
    reload_seen = set()

    def new_seg(q, k, kind):
        cur[q], A[q] = len(segs), I1
        segs.append((q, k, kind))
        return cur[q]

    def reload_point(q, k):
        """Under PESSIMISTIC a fresh |0> may be loaded into q's slot after every operation: it must be a + channel
        eigenstate in the frame.  Reload markers add nothing, since every operation is a loss location."""
        key = (cur[q], str(A[q]))
        if key not in reload_seen:
            reload_seen.add(key)
            uses.append((cur[q], A[q], "Z", f"reload of qubit {q} at instruction {k}"))
            checks.append(Prep(cur[q], A[q], "Z", 1, k, q, True))

    m = 0
    for q in range(n):                             # the initial |0> is checked as a reload before instruction 0
        new_seg(q, -1, "start")
        reload_point(q, -1)
    for k, inst in enumerate(ops):
        name, targets = inst.name, inst.targets_copy()
        qs = [t.value for t in targets]
        if name == "MPAD":
            for t in targets:
                kappa[m] = bool(t.value)
                m += 1
        elif _frame_free(inst):
            pass
        elif name in mf.UNITARY_1Q:
            g = stim.Tableau.from_named_gate(name)
            for q in qs:
                A[q] = g * A[q]
                reload_point(q, k)
        elif name in mf.RESET:
            for q in qs:
                s = new_seg(q, k, "reset")
                uses.append((s, I1, mf.RESET[name], f"reset at instruction {k}"))
                checks.append(Prep(s, I1, mf.RESET[name], 1, k, q, False))
                reload_point(q, k)
        elif name in mf.MEAS:
            ax = mf.MEAS[name]
            for t in targets:
                q = t.value
                old, A_old = cur[q], A[q]
                uses.append((old, A_old, ax, f"readout at instruction {k}"))
                meas.append((m, old, A_old, ax, t.is_inverted_result_target))
                m += 1
                s = new_seg(q, k, "reset" if name.startswith("MR") else "readout")
                uses.append((s, I1, ax, f"state after instruction {k}"))
                if name.startswith("MR"):
                    checks.append(Prep(s, I1, ax, 1, k, q, False))
                else:                               # the readout leaves its eigenstate; sign = the record
                    checks.append(Link(old, A_old, s, ax, k, q))
                reload_point(q, k)
        elif name in mf.UNITARY_2Q:
            shape = SHAPES[name]
            for a, b in zip(qs[::2], qs[1::2]):
                sa, sb, Aa, Ab = cur[a], cur[b], A[a], A[b]
                where = f"{name} at instruction {k} on {(a, b)}"
                if shape[0] == "free":
                    continue
                if shape[0] == "none":
                    raise _Fail(f"{where}: CSS in no local frame")
                if shape[0] == "ctrl":
                    _, P, Q = shape
                    uses += [(sa, Aa, P, where), (sb, Ab, Q, where)]
                    ctrls.append(((sa, Aa, P), (sb, Ab, Q), where))
                else:
                    _, M0, need = shape
                    perms.append((sa, sb, Ab.inverse() * M0 * Aa, where))
                    uses += [(sa, Aa, ax, where) for ax in need]
                checks.append(Gate(name, sa, Aa, sb, Ab, k, (a, b)))
    res.segments = len(segs)

    # ---------------------------------------------------------------- 2. permutation gates: weighted union-find
    parent, T, size = list(range(len(segs))), [I1] * len(segs), [1] * len(segs)

    def find(x):
        """(root, T) with frame(x) = T frame(root); compresses the path."""
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        to_root = I1
        for y in reversed(path):
            to_root = T[y] * to_root
            parent[y], T[y] = x, to_root
        return x, (T[path[0]] if path else I1)

    for sa, sb, M, why in perms:
        (ra, Ta), (rb, Tb) = find(sa), find(sb)
        if ra != rb:
            link = Tb.inverse() * M * Ta                         # frame(rb) = link frame(ra)
            if size[ra] < size[rb]:
                ra, rb, link = rb, ra, link.inverse()
            parent[rb], T[rb], size[ra] = ra, link, size[ra] + size[rb]
        elif not _is_pauli((M * Ta).inverse() * Tb):
            raise _Fail(f"{why}: closes a loop that needs a non-Pauli frame change")

    def root_axis(s, A_, axis):
        r, Tl = find(s)
        return r, image(A_ * Tl, axis)[0]

    # ---------------------------------------------------------------- 3. axes per group, then channel roles
    axes, why_axis = {}, {}
    for s, A_, axis, why in uses:
        r, letter = root_axis(s, A_, axis)
        axes.setdefault(r, set()).add(letter)
        why_axis.setdefault(r, []).append(why)
    for r, d in axes.items():
        if len(d) > 2:
            q, k, kind = segs[r]
            raise _Fail(f"qubit {q} (segment from {kind} at {k}) needs all three axes: {why_axis[r][:4]}")
    first = {r: min(d) for r, d in axes.items()}
    pp, par, psize = {}, {}, {}

    def pfind(x):
        """(component, beta): beta = 1 iff the group's first axis carries the Z channel, relative to the component."""
        path = []
        while pp.setdefault(x, x) != x:
            path.append(x)
            x = pp[x]
        to_root = 0
        for y in reversed(path):
            to_root ^= par[y]
            pp[y], par[y] = x, to_root
        return x, (par[path[0]] if path else 0)

    for (sa, Aa, P), (sb, Ab, Q), why in ctrls:
        (ra, la), (rb, lb) = root_axis(sa, Aa, P), root_axis(sb, Ab, Q)
        c = (la != first[ra]) ^ (lb != first[rb]) ^ 1          # beta_a xor beta_b = c
        (x, sx), (y, sy) = pfind(ra), pfind(rb)
        if x == y:
            if sx ^ sy != c:
                raise _Fail(f"{why}: conflicting channel roles")
        else:
            if psize.get(x, 1) < psize.get(y, 1):
                x, y = y, x
            pp[y], par[y], psize[x] = x, sx ^ sy ^ c, psize.get(x, 1) + psize.get(y, 1)
    res.groups = len({find(s)[0] for s in range(len(segs))})

    def frame_class(r):
        """Frame class of root r: its axes' channels follow its role.  A root with fewer than two axes takes any
        free second axis; no rule reads it, and flipping a whole component's roles conjugates every rule by H."""
        d = axes.get(r)
        if not d:
            return 0
        _, beta = pfind(r)
        f = first[r]
        want = {f: "Z" if beta else "X"}
        other = [a for a in d if a != f]
        want[other[0] if other else OTHER[f]] = "X" if want[f] == "Z" else "Z"
        (c,) = [c for c in range(6) if all(channel(c, a) == ch for a, ch in want.items())]
        return c

    cls_of, cache = {}, {}

    def frame0(s, A_):
        """(root, frame at time A_ of segment s, without the root's Pauli part)."""
        r, Tl = find(s)
        if r not in cls_of:
            cls_of[r] = frame_class(r)
        return r, A_ * Tl * REPS[cls_of[r]]

    # ---------------------------------------------------------------- 4. signs: one affine system over the Pauli parts
    system = _GF2()
    for chk in checks:
        roots, eqs = _sign_equations(chk, frame0, cache)
        if not all(system.add(*row) for row in _place(eqs, roots)):
            raise _Fail(f"{_describe(chk)}: no choice of Pauli parts gives it a + sign together with the checks before it")
    sol = system.solution()

    def frame(s, A_):
        r, F = frame0(s, A_)
        return F * PAULI_PART[((sol >> (2 * r)) & 1) | (((sol >> (2 * r + 1)) & 1) << 1)]

    # ---------------------------------------------------------------- 5. kappa
    for mi, s, A_, axis, inverted in meas:
        letter, sign = image(frame(s, A_), axis)
        assert letter in "XZ", (mi, letter)
        kappa[mi] = (sign < 0) ^ inverted
    res.reference = kappa
    res.verdict = "frame-css"


def _describe(chk):
    if isinstance(chk, Prep):
        what = "reload" if chk.reload else "prepared state"
        return f"{what} of qubit {chk.q} at instruction {chk.k}"
    if isinstance(chk, Link):
        return f"readout of qubit {chk.q} at instruction {chk.k} (sign carried into the next segment)"
    return f"{chk.name} at instruction {chk.k} on {chk.qs}"


# ------------------------------------------------------------------ signs
def _sign_equations(chk, frame0, cache):
    """(roots, affine equations over the roots' Pauli parts) under which the check holds with + signs."""
    if isinstance(chk, Prep):
        r, F = frame0(chk.seg, chk.A)
        key = ("prep", str(F), chk.axis, chk.sign)
        if key not in cache:
            cache[key] = _equations(lambda p: image(F * PAULI_PART[p[0]], chk.axis, chk.sign) in (("X", 1), ("Z", 1)), 1)
        return [r], cache[key]
    if isinstance(chk, Link):
        (ro, Fo), (rn, Fn) = frame0(chk.old, chk.A), frame0(chk.new, I1)
        same = ro == rn
        key = ("link", str(Fo), str(Fn), chk.axis, same)
        if key not in cache:
            def ok(p):
                po, pn = (p[0], p[0]) if same else p
                return image(Fo * PAULI_PART[po], chk.axis)[1] == image(Fn * PAULI_PART[pn], chk.axis)[1]
            cache[key] = _equations(ok, 1 if same else 2)
        return ([ro] if same else [ro, rn]), cache[key]
    (ra, Fa), (rb, Fb) = frame0(chk.sa, chk.Aa), frame0(chk.sb, chk.Ab)
    same = ra == rb
    key = ("gate", chk.name, str(Fa), str(Fb), same)
    if key not in cache:
        G = stim.Tableau.from_named_gate(chk.name)

        def ok(p):
            pa, pb = (p[0], p[0]) if same else p
            W = (Fa * PAULI_PART[pa]) + (Fb * PAULI_PART[pb])
            return positive_css(W.inverse() * G * W)
        cache[key] = _equations(ok, 1 if same else 2)
    return ([ra] if same else [ra, rb]), cache[key]


def _is_pauli(t):
    return all(str(t.x_output(k))[1:] == "_" * k + "X" + "_" * (len(t) - k - 1) and
               str(t.z_output(k))[1:] == "_" * k + "Z" + "_" * (len(t) - k - 1) for k in range(len(t)))


def _equations(pred, nroots):
    """Affine equations (mask over 2 * nroots bits, rhs) whose solutions are {p : pred(p)}, p = one Pauli part
    per root (bits x + 2 z).  A sign condition is affine in the Pauli parts (asserted)."""
    nb = 2 * nroots
    good = [v for v in range(1 << nb) if pred([(v >> (2 * i)) & 3 for i in range(nroots)])]
    if not good:
        return [(0, 1)]
    eqs = []
    for f in range(1, 1 << nb):
        vals = {(f & v).bit_count() & 1 for v in good}
        if len(vals) == 1:
            eqs.append((f, vals.pop()))
    assert sorted(v for v in range(1 << nb) if all((f & v).bit_count() & 1 == c for f, c in eqs)) == good
    return eqs


def _place(eqs, roots):
    out = []
    for f, c in eqs:
        mask = 0
        for i, r in enumerate(roots):
            mask |= ((f >> (2 * i)) & 3) << (2 * r)
        out.append((mask, c))
    return out


class _GF2:
    """Incremental GF(2) elimination over int bitmasks: add() refuses an equation that contradicts the others."""

    def __init__(self):
        self.piv = {}

    def add(self, y, v) -> bool:
        while y:
            h = y.bit_length() - 1
            if h not in self.piv:
                self.piv[h] = (y, v)
                return True
            py, pv = self.piv[h]
            y, v = y ^ py, v ^ pv
        return not v

    def solution(self) -> int:
        r = 0
        for h in sorted(self.piv):
            y, v = self.piv[h]
            if ((y & ~(1 << h) & r).bit_count() & 1) ^ v:
                r |= 1 << h
        return r
