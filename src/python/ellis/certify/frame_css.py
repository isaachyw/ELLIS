"""Frame-CSS: is a circuit CSS after relabelling each atom life by a single-qubit Clifford?

A *life* of a qubit runs from a reset (or the circuit start, an implicit |0>) to the next reset; MR/MRX/MRY
end one life and start the next.  Each life l gets one unknown single-qubit Clifford U_l.  Its label at
time t is U(t) = A_t U_l, where A_t is the product of the life's single-qubit gates so far, so a label
changes only across single-qubit gates (by exactly that gate) and never across a two-qubit gate.  The
frame image of a physical Pauli P is tau(P) = U(t)^dag P U(t).  The circuit is *frame-CSS* if some labels give:

  1. every reset of axis P (and every life start):   tau(P) in {+Z, +X};
  2. every measurement of axis P:                     tau(P) in {+Z, -Z, +X, -X}  (kappa_m = 1 iff minus);
  3. every controlled-Pauli gate C(A, B) (CX, CY, CZ, XCZ, ...):  {tau_a(A), tau_b(B)} = {+Z, +X},
     i.e. in the frame it is exactly a CNOT, the two operands in opposite roles;
  4. every SWAP(a, b): equal labels on both operands.

Theorem (exact level).  If the circuit is frame-CSS, the plain masked frame with reference kappa is exact
for every loss pattern: losses after any operation, any number of them, with or without reload, with
noise and Pauli gates in the frame.  Proof: relabelling is an identity of processes; in the frame every
lossy circuit is a CSS circuit whose resets are + resets (a lost atom is a partial trace plus dropped gates,
and the subgroup of a + CSS stabilizer group that is trivial on a qubit is again + CSS), so every
deterministic outcome reads 0 in the frame, i.e. kappa (plus inversions and MPAD values) physically.

Class level (signs ignored).  If labels exist only up to Paulis, each frame operation is (Pauli) x (exact
CSS operation); the leftover Paulis go into the masked frame as a gauge, each masked by the alive bits of
all operands of its operation, and the same reference kappa is exact for every loss pattern.

Decision procedure (near-linear): a weighted union-find over lives for the SWAP equalities, per-group axis
and sign bookkeeping (a group may use at most two Pauli axes), and a parity union-find for the CNOT
roles.  Ported from the conditions study (conditions_scratch/relabel/framecss.py), with inverted measurement
targets and MPAD values now included in kappa and unsupported operations reported instead of misparsed.
"""
from dataclasses import dataclass, field

import numpy as np
import stim

from ellis import frame as mf

CTRL = {"CX": ("Z", "X"), "CY": ("Z", "Y"), "CZ": ("Z", "Z"), "XCX": ("X", "X"), "XCY": ("X", "Y"),
        "XCZ": ("X", "Z"), "YCX": ("Y", "X"), "YCY": ("Y", "Y"), "YCZ": ("Y", "Z")}
RESET_AXIS = {"R": "Z", "RX": "X", "RY": "Y"}
MEAS_AXIS = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
I1 = stim.Tableau(1)


def _all_cliffords():
    found, frontier = {}, [I1]
    while frontier:
        t = frontier.pop()
        key = (str(t.x_output(0)), str(t.z_output(0)))
        if key not in found:
            found[key] = t
            frontier += [stim.Tableau.from_named_gate(g) * t for g in ("H", "S")]
    return list(found.values())


CLIFFORDS = _all_cliffords()          # the 24 single-qubit Cliffords (mod global phase)


def _frame(U, axis, sign=1):
    """U^dag (sign * axis) U as (axis letter, sign)."""
    p = U.inverse()(stim.PauliString(axis)) * sign
    return "_XYZ"[p[0]], int(p.sign.real)


def _is_pauli(t):
    n = len(t)
    return all(str(t.x_output(k))[1:] == "_" * k + "X" + "_" * (n - k - 1) and
               str(t.z_output(k))[1:] == "_" * k + "Z" + "_" * (n - k - 1) for k in range(n))


def _pauli_of(t):
    """The Pauli P whose conjugation is the Pauli tableau t."""
    out = stim.PauliString(len(t))
    for k in range(len(t)):
        sx, sz = str(t.x_output(k))[0] == "-", str(t.z_output(k))[0] == "-"
        out[k] = {(False, False): 0, (True, False): 3, (False, True): 1, (True, True): 2}[(sx, sz)]
    return out


@dataclass
class Relabeling:
    frame_css: bool                    # labels exist, at least up to Paulis (class level)
    exact: bool                        # labels exist with every sign right: no gauge needed
    reference: np.ndarray | None = None          # kappa, the record for frame.sample(reference=...)
    gauge: dict = field(default_factory=dict)    # {index: [(qubit, pauli, mask_qubits)]} for sample(gauge=...)
    lives: int = 0
    labels: list = field(default_factory=list)   # U_l of each life at its start (frame map P -> U_l^dag P U_l)
    reasons: list = field(default_factory=list)  # why not frame-CSS / not exact


def _parse(circuit):
    """Lives and events (resets, measurements, controlled-Pauli gates, SWAPs) in time order.
    Pauli gates, noise and loss live in the masked frame and are skipped."""
    n = circuit.num_qubits
    life, A, lives, events, m = {}, {}, [], [], 0

    def new_life(q, k, axis):
        life[q], A[q] = len(lives), I1
        lives.append((q, k))
        events.append({"kind": "reset", "k": k, "l": life[q], "q": q, "A": I1, "axis": axis})

    for q in range(n):
        new_life(q, -1, "Z")
    for k, inst in enumerate(circuit.flattened()):
        name, gd, targets = inst.name, stim.gate_data(inst.name), inst.targets_copy()
        if name == "MPAD":
            for t in targets:
                events.append({"kind": "pad", "m": m, "value": t.value}); m += 1
            continue
        if name in mf.ANNOTATIONS or name in mf.PAULI or (gd.is_noisy_gate and not gd.produces_measurements):
            continue
        if any(not t.is_qubit_target for t in targets):
            raise ValueError(f"instruction {k} `{inst}`: only plain qubit targets are supported")
        qs = [t.value for t in targets]
        if name in mf.UNITARY_1Q:
            g = stim.Tableau.from_named_gate(name)
            for q in qs: A[q] = g * A[q]
        elif name in RESET_AXIS:
            for q in qs: new_life(q, k, RESET_AXIS[name])
        elif name in MEAS_AXIS:
            for t in targets:
                q = t.value
                events.append({"kind": "meas", "k": k, "l": life[q], "q": q, "A": A[q], "axis": MEAS_AXIS[name],
                               "m": m, "inverted": t.is_inverted_result_target}); m += 1
                if name.startswith("MR"): new_life(q, k, MEAS_AXIS[name])
        elif name in CTRL or name == "SWAP":
            for a, b in zip(qs[::2], qs[1::2]):
                events.append({"kind": "swap" if name == "SWAP" else "ctrl", "name": name, "k": k, "q": (a, b),
                               "l": (life[a], life[b]), "A": (A[a], A[b]), "axis": CTRL.get(name)})
        else:
            raise ValueError(f"instruction {k} `{inst}`: `{name}` has no frame-CSS form")
    return lives, events, m


def relabel(circuit: stim.Circuit) -> Relabeling:
    """Decide frame-CSS (exact and class level) and build the reference kappa and the gauge."""
    try:
        lives, events, n_meas = _parse(circuit)
    except ValueError as e:
        return Relabeling(False, False, reasons=[str(e)])
    res = Relabeling(True, True, lives=len(lives))

    # 1. SWAP groups: A_a U_a = A_b U_b at the SWAP, so U_b = (A_b^-1 A_a) U_a; weighted union-find U_l = T_l U_root.
    parent, T = list(range(len(lives))), [I1] * len(lives)

    def find(x):
        if parent[x] == x:
            return x, I1
        r, Tp = find(parent[x])
        parent[x], T[x] = r, T[x] * Tp
        return r, T[x]

    for e in events:
        if e["kind"] != "swap": continue
        (la, lb), (Aa, Ab) = e["l"], e["A"]
        M = Ab.inverse() * Aa
        (ra, Ta), (rb, Tb) = find(la), find(lb)
        if ra != rb:
            parent[rb], T[rb] = ra, Tb.inverse() * M * Ta
        elif (D := (M * Ta).inverse() * Tb) != I1:
            res.exact = False
            if not _is_pauli(D):
                res.frame_css = False
                res.reasons.append(f"SWAP loop at instruction {e['k']} needs a non-Pauli label change")
            else:
                res.reasons.append(f"SWAP loop at instruction {e['k']} needs a Pauli label change (gauge)")

    # 2. Express every use in its group's root frame: (axis, sign) before applying the unknown U_root.
    axes = {}                                    # root -> {axis: set of signs of the signed uses}

    def use(l, A, axis, signed):
        r, Tl = find(l)
        ax, s = _frame(A * Tl, axis)
        axes.setdefault(r, {}).setdefault(ax, set())
        if signed: axes[r][ax].add(s)
        return r, ax

    for e in events:
        if e["kind"] == "reset":
            e["use"] = [use(e["l"], e["A"], e["axis"], True)]
        elif e["kind"] == "meas":
            e["use"] = [use(e["l"], e["A"], e["axis"], False)]
        elif e["kind"] == "ctrl":
            e["use"] = [use(e["l"][i], e["A"][i], e["axis"][i], True) for i in (0, 1)]
    for r, d in axes.items():
        q, k = lives[r]
        if len(d) > 2:
            res.frame_css = res.exact = False
            res.reasons.append(f"qubit {q} (life from instruction {k}) needs all three axes X, Y, Z")
        elif any(len(s) > 1 for s in d.values()):
            res.exact = False
            res.reasons.append(f"qubit {q} (life from instruction {k}) uses an axis with both signs (gauge)")
    if not res.frame_css:
        return res

    # 3. Roles: bit beta_r = "the group's first axis is frame-Z"; every C(A,B) needs opposite roles.
    first = {r: min(d) for r, d in axes.items()}
    pp, par = {}, {}

    def pfind(x):
        if pp.setdefault(x, x) == x:
            par.setdefault(x, 0)
            return x, 0
        r, s = pfind(pp[x])
        pp[x] = r; par[x] ^= s
        return r, par[x]

    for e in events:
        if e["kind"] != "ctrl": continue
        (ra, axa), (rb, axb) = e["use"]
        c = (axa != first[ra]) ^ (axb != first[rb]) ^ 1          # beta_a xor beta_b = c
        (x, sx), (y, sy) = pfind(ra), pfind(rb)
        if x == y:
            if sx ^ sy != c:
                res.frame_css = res.exact = False
                res.reasons.append(f"{e['name']} at instruction {e['k']} on qubits {e['q']}: conflicting CNOT roles")
                return res
        else:
            pp[y], par[y] = x, sx ^ sy ^ c

    # 4. Build one solution U_root per group, then kappa and the gauge.
    U_root = {}
    for r, d in axes.items():
        beta = pfind(r)[1]
        roles = {ax: ("Z" if (beta if ax == first[r] else not beta) else "X", min(d[ax]) if d[ax] else 1) for ax in d}
        if len(roles) == 1:
            (ax, (role, _)), = roles.items()
            roles[{"X": "Z", "Z": "X", "Y": "Z"}[ax]] = ("X" if role == "Z" else "Z", 1)
        cands = [U for U in CLIFFORDS if all(_frame(U, ax) == (role, s) for ax, (role, s) in roles.items())]
        assert len(cands) == 1, (roles, len(cands))
        U_root[r] = cands[0]

    def label(l, A):
        r, Tl = find(l)
        return A * Tl * U_root.get(r, I1)

    res.labels = [label(l, I1) for l in range(len(lives))]

    kappa = np.zeros(n_meas, bool)
    for e in events:
        if e["kind"] == "pad":
            kappa[e["m"]] = bool(e["value"])
        elif e["kind"] == "meas":
            ax, s = _frame(label(e["l"], e["A"]), e["axis"])
            assert ax in "XZ"
            kappa[e["m"]] = (s < 0) ^ e["inverted"]
        elif e["kind"] == "reset":
            U = label(e["l"], e["A"])
            ax, s = _frame(U, e["axis"])
            assert ax in "XZ"
            if s < 0:                                  # a '-' reset (or start, k = -1): fix it right after
                fix = U(stim.PauliString("Z" if ax == "X" else "X"))
                res.gauge.setdefault(e["k"] + 1, []).append((e["q"], "_XYZ"[fix[0]], (e["q"],)))
        elif e["kind"] in ("ctrl", "swap"):
            a, b = e["q"]
            W = label(e["l"][0], e["A"][0]) + label(e["l"][1], e["A"][1])
            G = W.inverse() * stim.Tableau.from_named_gate(e["name"]) * W     # the gate in the frame
            cx, swap = stim.Tableau.from_named_gate("CX"), stim.Tableau.from_named_gate("SWAP")
            for R in (cx, swap * cx * swap, swap):
                P = G * R.inverse()                    # G = P R: the Pauli part P acts right after R
                if _is_pauli(P):
                    break
            else:
                raise AssertionError(f"{e['name']} at instruction {e['k']} is not CSS in the frame")
            phys = W(_pauli_of(P))
            for i, q in enumerate((a, b)):
                if phys[i]:
                    res.gauge.setdefault(e["k"] + 1, []).append((q, "_XYZ"[phys[i]], (a, b)))
    if res.gauge:
        res.exact = False
    res.reference = kappa
    return res
