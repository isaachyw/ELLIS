"""Closed S/CZ diagonal blocks: a static common-reference proof beyond frame-CSS (paper App. B)."""

from dataclasses import dataclass, field

import numpy as np
import stim

from ellis import frame as mf

# ------------------------------------------------------------------ closed diagonal blocks (H3)
_FRAME_ONLY = mf.PAULI | {"I", "II"}


def _frame_only(inst) -> bool:
    gd = stim.gate_data(inst.name)
    return inst.name in mf.ANNOTATIONS or inst.name in _FRAME_ONLY or (gd.is_noisy_gate and not gd.produces_measurements)


@dataclass
class DiagonalBlocks:
    proven: bool
    reference: np.ndarray | None = None
    gauge: dict = field(default_factory=dict)
    blocks: int = 0
    reasons: list = field(default_factory=list)


def diagonal_blocks(circuit: stim.Circuit) -> DiagonalBlocks:
    """Closed diagonal blocks: every S / S_DAG / CZ lies in a block and everything else is CSS.

    On each qubit, a *run* is a maximal stretch of S, S_DAG and CZ gates between two other operations (a
    Pauli X-part is constant along a run); runs joined by a CZ form a block, whose product U_B of
    commuting diagonal gates is computed exactly (S counts mod 4 per qubit, CZ counts mod 2 per pair).

    Theorem (corrected H3 of the conditions study).  If every block has U_B proportional to I, and every
    other operation is CSS (Z/X resets and measurements, CX, SWAP, H only right after a reset or right
    before a measurement), then every deterministic parity reads 0 in every lossy circuit, so the trivial
    reference (inverted readouts and MPAD values only) is exact for every loss pattern.  If U_B is
    proportional to Z^c, the same holds with a masked Z gauge on c.  Proof: a lost atom can carry no X-part
    through its run (it has no later readout there and diagonal gates do not move X-parts), so the block
    gates that loss removes would have contributed no phase; the block's total phase on any Pauli with
    X-part x is that of U_B, i.e. 0 (or 2 c.x).  The rest is CSS, which adds no phase."""
    flat = list(circuit.flattened())
    res = DiagonalBlocks(True)
    run, n_runs, run_qubit = {}, 0, []  # current run id per qubit; the qubit of each run
    member = {}  # (instruction, qubit) -> run id
    for k, inst in enumerate(flat):
        if _frame_only(inst):
            continue
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in ("S", "S_DAG", "CZ"):
            for q in qs:
                if q not in run:
                    run[q] = n_runs
                    n_runs += 1
                    run_qubit.append(q)
                member[(k, q)] = run[q]
        else:
            for q in qs:
                run.pop(q, None)
    # A run's X-part is constant, so the block phase is a diagonal Clifford on the RUNS (not on qubits:
    # one qubit can have two runs in a block, with different X-parts): S counts mod 4 per run, CZ parity
    # per pair of runs.  Blocks (runs joined by CZ) are only counted.
    parent = list(range(n_runs))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    s_count, cz_count, last_gate = {}, {}, {}
    for k, inst in enumerate(flat):
        if inst.name in ("S", "S_DAG"):
            for t in inst.targets_copy():
                r = member[(k, t.value)]
                s_count[r] = (s_count.get(r, 0) + (1 if inst.name == "S" else 3)) % 4
                last_gate[r] = k
        elif inst.name == "CZ":
            qs = [t.value for t in inst.targets_copy()]
            for a, b in zip(qs[::2], qs[1::2]):
                ra, rb = member[(k, a)], member[(k, b)]
                parent[find(ra)] = find(rb)
                cz_count[(min(ra, rb), max(ra, rb))] = cz_count.get((min(ra, rb), max(ra, rb)), 0) ^ 1
    res.blocks = len({find(r) for r in range(n_runs)})
    for (ra, rb), odd in cz_count.items():
        if odd:
            res.proven = False
            res.reasons.append(f"odd number of CZ between qubits {run_qubit[ra]} and {run_qubit[rb]} in one block")
    for r, cnt in s_count.items():
        if cnt % 2:
            res.proven = False
            res.reasons.append(f"net S-count {cnt} mod 4 on qubit {run_qubit[r]} in one block")
        elif cnt == 2:  # the block is ~Z there: masked Z gauge inside the run
            res.gauge.setdefault(last_gate[r] + 1, []).append((run_qubit[r], "Z", (run_qubit[r],)))
    # everything outside the blocks must be CSS
    prev, nxt = {}, {}
    ops = [
        (k, inst.name, [t.value for t in inst.targets_copy() if t.is_qubit_target])
        for k, inst in enumerate(flat)
        if not _frame_only(inst) and inst.name not in ("S", "S_DAG", "CZ")
    ]
    for i, (k, name, qs) in enumerate(ops):
        for q in qs:
            if q in prev:
                nxt[(prev[q][0], q)] = name
            prev[q] = (k, name)
    seen = {}
    for k, name, qs in ops:
        for q in qs:
            before = seen.get(q, "R")  # the implicit start is a reset to |0>
            seen[q] = name
            if name in ("R", "RX", "M", "MX", "MR", "MRX", "CX", "SWAP"):
                continue
            if name == "H" and (before in ("R", "RX", "MR", "MRX") or nxt.get((k, q)) in ("M", "MX", "MR", "MRX")):
                continue
            res.proven = False
            res.reasons.append(f"`{name}` at instruction {k} on qubit {q} is outside the CSS rules")
            break
    if res.proven:
        kappa = np.zeros(circuit.num_measurements, bool)
        m = 0
        for inst in flat:
            if inst.name == "MPAD" or stim.gate_data(inst.name).produces_measurements:
                for t in inst.targets_copy():
                    kappa[m] = t.value if inst.name == "MPAD" else t.is_inverted_result_target
                    m += 1
        res.reference = kappa
    return res
