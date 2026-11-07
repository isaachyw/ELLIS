"""certify(): frame-CSS proofs under a loss model, contradiction witnesses, exact sign constraints, ghost-equivalent
loss slots."""
from pathlib import Path

import numpy as np
import stim
from ellis.certify import (LossModel, certify, check_reference, constraints, frame_css, loss_family, loss_locations,
                           loss_slots, representative_slots, search)
from ellis.check_exact import same_distribution
from ellis.gf2 import Span, to_ints

from ellis import frame as mf

LIGHTSTIM_CIRCUITS = Path(__file__).resolve().parents[1] / "fixtures" / "lightstim"


def parity(x):
    return x.bit_count() & 1


def pattern_space(circ, losses, reload, seed, reference):
    """(V_s span, masked-frame sample a_s, exact Stim sample b_s) of the lossy circuit C_s."""
    n = circ.num_measurements + 64
    a, _ = mf.sample(circ, n, seed=seed, reload=reload, forced_losses=losses, reference=reference, canonical=False)
    b = mf.explicit_lossy_circuit(circ, losses, reload).compile_sampler(seed=seed).sample(n)
    A, B = to_ints(a), to_ints(b)
    V = Span()
    for r in B[1:]:
        V.add(r ^ B[0])
    return V, A[0], B[0]


def rank(vectors):
    piv = {}
    for v in vectors:
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = v; break
            v ^= piv[h]
    return len(piv)


def test_css_memory_is_proven_with_zero_reference():
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                               after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3)
    cert = certify(c)
    assert (cert.verdict, cert.method) == ("proven", "frame-css")
    assert cert.reference is not None and not cert.reference.any()


def test_cz_coupling_is_frame_css():
    # a CZ is a CNOT once qubit 1 is relabelled by H (the XZZX mechanism)
    cert = certify(stim.Circuit("RX 0\nR 1\nCZ 0 1\nMX 0\nM 1"))
    assert cert.verdict == "proven"


EVERY = LossModel(after="any", noiseless_loss_free=False, from_start=True)


def test_every_loss_location_on_static_proofs():
    # losses after every operation and before the first: the spot check samples that family
    cert = certify(stim.Circuit("RX 0\nR 1\nCZ 0 1\nMX 0\nM 1"), EVERY)
    assert (cert.verdict, cert.method, cert.family_size > 0) == ("proven", "frame-css", True)
    cert = certify(stim.Circuit("RX 0 1 2\nCZ 0 1 1 2 0 2\nCZ 0 1 1 2 0 2\nMX 0 1 2"), EVERY)
    assert (cert.verdict, cert.method, cert.family_size > 0) == ("proven", "diagonal-blocks", True)


def test_benchmark_xzzx_is_proven():
    f = min(LIGHTSTIM_CIRCUITS.glob("*xzzx_surface_code_Z*"))
    assert certify(stim.Circuit(f.read_text())).verdict == "proven"


def test_kappa_differs_from_stim_reference():
    c = stim.Circuit("SQRT_Y_DAG 0\nRX 1\nS 1\nSQRT_X 2\nCX 1 2\nCZ 1 0\nMX 0\nM 1\nMY 2")
    rel = frame_css(c, EVERY)
    assert rel.proven and rel.exact and rel.reference is not None
    assert rel.reference.astype(int).tolist() == [1, 0, 1]
    fam = loss_family(c, slots=loss_locations(c, EVERY), n_random=0)
    assert check_reference(c, rel.reference, fam) == []
    assert check_reference(c, mf.strip_loss(c).reference_sample(), fam)   # Stim's own reference fails


def test_contradiction_found_minimal_and_real():
    c = stim.Circuit("RX 2\nCX 2 0\nCZ 0 2\nCX 2 0\nMX 1\nCX 1 2\nMX 1")
    cert = certify(c, n_random=0)
    assert cert.verdict == "contextual" and cert.witness is not None
    assert len(cert.witness) == 2
    total_y, total_v = np.zeros(c.num_measurements, int), 0
    for pattern, idx, val in cert.witness:
        # independent check with Stim sampling: the parity is fixed to `val` in that lossy circuit
        samples = mf.explicit_lossy_circuit(c, pattern).compile_sampler(seed=1).sample(300)
        assert (samples[:, idx].sum(axis=1) % 2 == val).all()
        total_y[idx] ^= 1; total_v ^= val
    assert not total_y.any() and total_v == 1                # the parities cancel, the values add to 1
    assert search(c, [w[0] for w in cert.witness])[1] is not None


def test_constraints_match_sampling():
    for name in ("rotated_surface_code_Z", "logical_s_roundtrip"):
        c = stim.Circuit(min(LIGHTSTIM_CIRCUITS.glob(f"*{name}*")).read_text()).flattened()
        m = c.num_measurements; nb = 8 * ((m + 7) // 8)
        packed = lambda y, m=m, nb=nb: sum(1 << (nb - 1 - j) for j in range(m) if (y >> j) & 1)
        for t in range(8):
            reload = bool(t % 2)
            s = loss_family(c, "any", reload, n_random=1, max_losses=3, seed=t)[-1]
            rows = constraints(c, s, reload)
            V, a0, b0 = pattern_space(c, s, reload, t, np.zeros(m, bool))
            assert all(parity(packed(y) & v) == 0 for y, _ in rows for v in V.piv.values())
            assert all(parity(packed(y) & (a0 ^ b0)) == val for y, val in rows)
            assert rank([y for y, _ in rows]) == m - V.dim


def test_long_records_with_nonzero_reference():
    # 70 inverted readouts: the reference is all ones, well past 64 bits
    c = stim.Circuit("R 0\n" + "M !0\n" * 70)
    rows = constraints(c, {})
    assert len(rows) == 70 and all(v == 1 for _, v in rows)
    assert check_reference(c, mf.strip_loss(c).reference_sample(), [{}]) == []
    assert check_reference(c, np.zeros(70, bool), [{}]) == [{}]


def test_ghost_equivalent_slots_collapse():
    c = stim.Circuit("R 0 1\nH 0\nS 0\nCX 0 1\nH 0\nCX 0 1\nM 0 1")
    reps = representative_slots(c, loss_slots(c, "any"))
    # qubit 0: losing it after R, H or S drops the same operations, as do losses after the first CX or H
    assert [k for k, q in reps if q == 0] == [0, 3, 5]


def test_closed_diagonal_blocks():
    s4 = stim.Circuit(min(LIGHTSTIM_CIRCUITS.glob("*logical_s_to_the_fourth*")).read_text())
    cert = certify(s4)
    assert (cert.verdict, cert.method) == ("proven", "diagonal-blocks")
    # a CZ triangle (odd cycle: not frame-CSS) applied twice is I, and S.S = Z leaves a Z on qubit 0:
    # proven with a masked Z gauge, and the gauged sampler is exact (qubit 0 ends in |->)
    c = stim.Circuit("RX 0 1 2\nCZ 0 1 1 2 0 2\nCZ 0 1 1 2 0 2\nS 0\nS 0\nMX 0 1 2")
    assert frame_css(c).exact              # losses only after the whole CZ layer: the triangles cancel in the prefix
    assert not frame_css(c, EVERY).proven  # a loss between the triangles' CZs
    cert = certify(c, EVERY)
    assert (cert.verdict, cert.method) == ("proven", "diagonal-blocks+gauge")
    m, _ = mf.sample(c, 100, reference=cert.reference, gauge=cert.gauge)
    assert (m == [1, 0, 0]).all()
    # a lone S is no closed block: not provable statically, and the exact search finds no contradiction
    cert = certify(stim.Circuit("RX 0 1\nCZ 0 1\nS 0\nCZ 0 1\nMX 0 1"))
    assert (cert.verdict, cert.method) == ("consistent_on_family", "exact-search")


def test_gate_shapes_beyond_controlled_paulis():
    # CXSWAP is a permutation gate: it ties the two frames.  ISWAP is CSS in no local frame.
    assert certify(stim.Circuit("R 0 1\nRX 2\nCX 2 0\nCXSWAP 0 1\nM 0 1\nMX 2")).verdict == "proven"
    rel = frame_css(stim.Circuit("RX 0\nR 1 2\nCX 0 2\nISWAP 0 1\nM 0 1 2"), EVERY)
    assert not rel.proven and "no local frame" in rel.reasons[0]


def test_readouts_start_new_segments():
    # qubit 0 is read, rotated by H and reused with the same partner: the readout starts a new frame
    c = stim.Circuit("RX 0\nR 1\nCX 0 1\nMX 0\nH 0\nCX 0 1\nM 0 1")
    rel = frame_css(c, EVERY)
    assert rel.proven and rel.exact
    assert check_reference(c, rel.reference, loss_family(c, slots=loss_locations(c, EVERY), n_random=50, seed=1)) == []


def test_noiseless_prefix_is_checked_as_a_state():
    # the noiseless encoder uses ISWAP, CSS in no frame, but leaves a Bell pair; losses start after it
    c = stim.Circuit("RX 0\nR 1 2\nCX[noiseless] 0 1\nISWAP[noiseless] 0 1\nISWAP_DAG[noiseless] 0 1\n"
                     "CX 0 2\nCX 1 2\nM 2\nMX 0 1")
    rel = frame_css(c)                                   # default model: losses after noisy two-qubit gates
    assert rel.proven and rel.exact and rel.prefix_end == 5
    fam = loss_family(c, slots=loss_locations(c, LossModel()), n_random=30, seed=2)
    assert check_reference(c, rel.reference, fam) == []
    assert not frame_css(c, EVERY).proven                # the encoder itself may lose atoms


def test_reload_markers():
    # gates H S S H (= X) then H S S H again: identity at the CX, but X halfway through
    run = "H 0\nTICK\nS 0\nTICK\nS 0\nTICK\nH 0\nTICK\n"
    base = "R 0 1\nRX 2\nCX 2 0\n" + run + "{marker}" + run + "CX 0 1\nM 0 1\nMX 2"
    plain = stim.Circuit(base.format(marker=""))
    assert frame_css(plain).exact                         # no reload: exact
    rel = frame_css(plain, LossModel(reload="anywhere"))  # a reload where the gates so far are X reads 1 at the CX
    assert not rel.proven and "reload of qubit 0" in rel.reasons[0]
    marked = stim.Circuit(base.format(marker="I_ERROR[reload] 0\n"))
    model = LossModel(after="any")
    rel = frame_css(marked, model)                        # the marker's reload: fixed by a masked gauge
    assert rel.proven and not rel.exact and rel.gauge
    for i, s in enumerate(loss_family(marked, slots=loss_locations(marked, model), n_random=40, max_losses=3, seed=3)):
        n = marked.num_measurements + 100
        a, _ = mf.sample(marked, n, seed=i, forced_losses=s, reference=rel.reference, gauge=rel.gauge)
        b = mf.explicit_lossy_circuit(marked, s).compile_sampler(seed=i).sample(n)
        assert same_distribution(a, b), s
