"""Terminal Pauli queries of the MSC fast path, checked against exact sign constraints for every absence mask.

A query gadget is accepted only as a star: basis rotations, then CXs from distinct controls into the readout
pivot, the readout, and the mirror image.  A control lost before the gadget drops both of its CXs, so its
factor leaves the query; a lost pivot makes the readout a coin.  A chain such as CX a->b, CX b->c, M c is not a
star: with b absent the physical query is Z_c, not Z_a Z_c, so deleting b's factor would be wrong, and the plan
must not accept it.  The oracle is check_reference (Stim flow generators of the explicit lossy circuit), which
shares no code with the fast path.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest
import stim

from ellis import _core
from ellis import frame as mf
from ellis.certify import check_reference
from ellis.msc.plan import _cultivation_plan, compile_plan
from ellis.msc.reference import CultivationReference

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cultivation"


def absent_before(circuit, boundary, qubits, earlier=()):
    """Losses with `qubits` already absent when the terminal gadgets start, plus (k, q) losses earlier on."""
    losses = {}
    for k, q in earlier:
        losses.setdefault(k, []).append(q)
    if qubits:
        losses.setdefault(boundary - 1, []).extend(qubits)
    return {k: sorted(set(v)) for k, v in losses.items()}


def star_circuit(rng, n, queries):
    """A cultivation-shaped circuit: R/RX, CXs that join every qubit to the S qubit, S, a CSS tail, then
    noiseless star gadgets.  queries: [(pivot, controls, rotations {qubit: 'H' | 'H_YZ'})]."""
    c = stim.Circuit()
    c.append("R", range(n))
    c.append("RX", [q for q in range(n) if rng.random() < 0.5] or [0])
    order = list(rng.permutation(n))
    for a, b in zip(order, order[1:]):
        c.append("CX", [int(a), int(b)])
    for _ in range(n):
        a, b = rng.choice(n, 2, replace=False)
        c.append("CX", [int(a), int(b)])
    c.append("S" if rng.random() < 0.5 else "S_DAG", [int(rng.integers(n))])
    c.append("TICK")
    for _ in range(n):
        a, b = rng.choice(n, 2, replace=False)
        c.append("CX", [int(a), int(b)])
    c.append("TICK")
    for pivot, controls, rotations in queries:
        compute = [(g, [q]) for q, g in rotations.items()] + [("CX", [a, pivot]) for a in controls]
        for g, qs in compute:
            c.append(g, qs, tag="noiseless")
        c.append("M", [pivot], tag="noiseless")
        for g, qs in reversed(compute):
            c.append(g, qs, tag="noiseless")
    return c


def random_queries(rng, n, n_queries):
    """Star queries with exactly one mixed query (the plan's condition)."""
    out, mixed_at = [], int(rng.integers(n_queries))
    for i in range(n_queries):
        size = int(rng.integers(2, min(n, 4) + 1))
        support = [int(q) for q in rng.choice(n, size, replace=False)]
        pivot, controls = support[0], support[1:]
        if i == mixed_at:                       # some but not all factors rotated, or a Y factor
            rot = {q: "H" for q in support if rng.random() < 0.5}
            if len(rot) in (0, len(support)):
                rot = {support[0]: "H_YZ"}
        else:                                   # pure Z, or pure X
            rot = {q: "H" for q in support} if rng.random() < 0.5 else {}
        out.append((pivot, controls, rot))
    return out


def gadget_start(circuit):
    """Index of the first noiseless operation: where the terminal gadgets begin."""
    return next(k for k, inst in enumerate(circuit.flattened())
                if inst.tag == "noiseless" and inst.name not in mf.ANNOTATIONS)


def assert_valid_for_every_mask(circuit, earlier_losses=((),)):
    """CultivationReference's record satisfies the exact constraints of every absence mask; returns its stats."""
    provider = CultivationReference(circuit)
    boundary = gadget_start(circuit)
    for earlier in earlier_losses:
        for r in range(circuit.num_qubits + 1):
            for mask in itertools.combinations(range(circuit.num_qubits), r):
                losses = absent_before(circuit, boundary, mask, earlier)
                ref = provider.get(losses)
                assert check_reference(circuit, ref, [losses]) == [], f"invalid reference for {losses}\n{circuit}"
    return provider.stats


@pytest.mark.parametrize("seed", range(40))
def test_star_queries_are_exact_for_every_absence_mask(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 6))
    circuit = star_circuit(rng, n, random_queries(rng, n, int(rng.integers(2, 4))))
    plan = _cultivation_plan(circuit)
    assert plan is not None and len(plan["queries"]) >= 2, circuit
    boundary = plan["boundary"]
    earlier = [()] + [((int(rng.integers(-1, boundary - 1)), int(rng.integers(n))),) for _ in range(2)]
    stats = assert_valid_for_every_mask(circuit, earlier)
    assert stats["queries"] == len(earlier) * 2**n


def test_star_queries_take_the_fast_path():
    # the exhaustive check above is only meaningful if the fast path answers most of those masks
    fast = total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 6))
        circuit = star_circuit(rng, n, random_queries(rng, n, int(rng.integers(2, 4))))
        stats = assert_valid_for_every_mask(circuit)
        fast, total = fast + stats["fast"], total + stats["queries"]
    assert fast > total // 2, (fast, total)


# The reviewer's counterexample with deterministic signs (the reference circuit has no Pauli gates, so the
# signs come from preparation): a = |-i> read through H_YZ (factor Y_a = -1), b = c = |0>.  All present, the
# gadget reads Y_a Z_b Z_c = -1.  With b absent both CXs drop and it reads Z_c = +1; deleting b's factor would
# predict Y_a Z_c = -1 instead.
CHAIN = """R 0 1 2
RX 0
S_DAG 0
TICK
H_YZ[noiseless] 0
CX[noiseless] 0 1
CX[noiseless] 1 2
M[noiseless] 2
CX[noiseless] 1 2
CX[noiseless] 0 1
H_YZ[noiseless] 0
"""


def test_chain_gadget_is_not_accepted_as_a_query():
    circuit = stim.Circuit(CHAIN)
    assert _cultivation_plan(circuit) is None
    meta = compile_plan(circuit)
    assert meta is None or meta["queries"] == []        # the general split has no terminal-query rule
    assert_valid_for_every_mask(circuit)


def test_deleting_an_absent_chain_factor_is_caught_by_the_oracle():
    # Force the chain through FastPlan as the query Y_a Z_b Z_c: the fast path then deletes b's factor when b
    # is absent, and the oracle must reject exactly that record while accepting the all-present one.
    circuit = stim.Circuit(CHAIN).flattened()
    boundary = gadget_start(circuit)
    plan = _core.FastPlan(mf.reference_skeleton(circuit), circuit.num_qubits, boundary, boundary, [0],
                          [(0, [0, 1, 2], "YZZ", 2, False)])
    record, fast = plan.reference({})
    assert fast and record == b"\x01"
    assert check_reference(circuit, np.frombuffer(record, dtype=np.bool_), [{}]) == []
    losses = absent_before(circuit, boundary, [1])
    record, fast = plan.reference(losses)
    assert fast and record == b"\x01"                   # Y_a Z_c = -1, the deleted-factor prediction
    assert check_reference(circuit, np.frombuffer(record, dtype=np.bool_), [losses]) == [losses]
    assert check_reference(circuit, np.zeros(1, bool), [losses]) == []      # the physical query Z_c = +1


def export_masks(queries, rng, small_queries=8):
    """Absence masks for an export's queries without enumerating subsets (one support has 29 qubits): for the
    largest query and a sample of the others, a few single absences, the whole support, and random subsets."""
    by_size = sorted(range(len(queries)), key=lambda i: len(queries[i][1]))
    chosen = [by_size[-1]] + [int(i) for i in rng.choice(by_size[:-1], small_queries, replace=False)]
    masks = []
    for i in chosen:
        support = list(queries[i][1])
        singles = rng.choice(support, min(len(support), 4), replace=False)
        masks += [(int(q),) for q in singles] + [tuple(support)]
        masks += [tuple(q for q in support if rng.random() < 0.5) for _ in range(2)]
    return list(dict.fromkeys(m for m in masks if m))


@pytest.mark.parametrize("distance", [3, 5])
def test_cultivation_exports_query_support_absences(distance):
    circuit = stim.Circuit.from_file(FIXTURES / f"msc_e2e_d{distance}_d15_r3_r5.stim")
    provider = CultivationReference(circuit)
    meta = provider.metadata
    assert meta is not None and meta["queries"], "the export should use the cultivation query plan"
    masks = export_masks(meta["queries"], np.random.default_rng(distance))
    assert len(masks) >= 30, len(masks)
    for mask in masks:
        losses = absent_before(circuit, meta["boundary"], mask)
        assert check_reference(circuit, provider.get(losses), [losses]) == [], losses
    assert provider.stats["fast"] > provider.stats["queries"] // 2, provider.stats
