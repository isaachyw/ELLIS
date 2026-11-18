"""certify(): frame-CSS proofs under the pessimistic loss model, contradiction witnesses, exact sign constraints, ghost-equivalent
loss slots."""
from pathlib import Path

import numpy as np
import pytest
import stim
from ellis.certify import (PESSIMISTIC, LossModel, certify, check_reference, constraints, frame_css, loss_family, loss_locations,
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
    assert cert.family_size == 0


def test_cz_coupling_is_frame_css():
    # a CZ is a CNOT once qubit 1 is relabelled by H (the XZZX mechanism)
    cert = certify(stim.Circuit("RX 0\nR 1\nCZ 0 1\nMX 0\nM 1"))
    assert cert.verdict == "proven"


@pytest.mark.parametrize("spot_check", [0, 50])
def test_every_loss_location_on_static_proofs(spot_check):
    # Static proofs cover every loss location; additional pattern checks are optional.
    cert = certify(stim.Circuit("RX 0\nR 1\nCZ 0 1\nMX 0\nM 1"), PESSIMISTIC, spot_check=spot_check)
    assert (cert.verdict, cert.method, cert.family_size > 0) == ("proven", "frame-css", bool(spot_check))


@pytest.mark.parametrize("model", [LossModel(), LossModel(after="any"), LossModel(reload="reset"), PESSIMISTIC])
def test_the_proof_ignores_the_model_and_holds_for_each(model):
    c = stim.Circuit("RY 0 1 2\nS 0\nCZ 2 0\nMX 0\nM 1\nMY 2")
    cert = certify(c, model)
    assert cert.verdict == "proven" and cert.reference.tolist() == frame_css(c).reference.tolist()
    fam = loss_family(c, reload=model.reload_at_reset, slots=loss_locations(c, model), pairs=True)
    assert check_reference(c, cert.reference, fam, model.reload_at_reset) == []


def test_benchmark_xzzx_is_proven():
    f = min(LIGHTSTIM_CIRCUITS.glob("*xzzx_surface_code_Z*"))
    assert certify(stim.Circuit(f.read_text())).verdict == "proven"


def test_kappa_differs_from_stim_reference():
    c = stim.Circuit("RY 0 1 2\nS 0\nCZ 2 0\nMX 0\nM 1\nMY 2")
    rel = frame_css(c)
    assert rel.verdict == "frame-css" and rel.reference is not None
    assert rel.reference.astype(int).tolist() == [1, 0, 0]
    fam = loss_family(c, slots=loss_locations(c, PESSIMISTIC), n_random=0)
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


def test_closed_s_blocks_are_no_proof_and_fall_back_to_the_search():
    # S S = Z between two CZs is identity up to a Pauli, but a loss or reload between them changes the circuit
    s4 = stim.Circuit(min(LIGHTSTIM_CIRCUITS.glob("*logical_s_to_the_fourth*")).read_text())
    assert certify(s4, n_random=0).method == "exact-search"
    cert = certify(stim.Circuit("RX 0 1\nCZ 0 1\nS 0\nCZ 0 1\nMX 0 1"))
    assert (cert.verdict, cert.method) == ("consistent_on_family", "exact-search")


def test_gate_shapes_beyond_controlled_paulis():
    # CXSWAP is a permutation gate: it ties the two frames.  ISWAP is CSS in no local frame.
    assert certify(stim.Circuit("R 0 1\nRX 2\nCX 2 0\nCXSWAP 0 1\nM 0 1\nMX 2")).verdict == "proven"
    rel = frame_css(stim.Circuit("RX 0\nR 1 2\nCX 0 2\nISWAP 0 1\nM 0 1 2"))
    assert not rel.proven and "no local frame" in rel.reasons[0]


def test_readouts_start_new_segments():
    # qubit 0 is read, rotated by H and reused with the same partner: the readout starts a new frame
    c = stim.Circuit("RX 0\nR 1\nCX 0 1\nMX 0\nH 0\nCX 0 1\nM 0 1")
    rel = frame_css(c)
    assert rel.proven
    assert check_reference(c, rel.reference, loss_family(c, slots=loss_locations(c, PESSIMISTIC), n_random=50, seed=1)) == []


def test_noiseless_operations_may_still_lose_atoms():
    # a reload after the noiseless H S reads |0> as a random X readout, so the tag must not exempt the run
    run = "H{t} 0\nS{t} 0\nS{t} 0\nH{t} 0\n"
    tagged = frame_css(stim.Circuit("R 0\n" + run.format(t="[noiseless]") + "M 0"))
    assert not tagged.proven and "reload of qubit 0" in tagged.reasons[0]
    assert tagged.reasons == frame_css(stim.Circuit("R 0\n" + run.format(t="") + "M 0")).reasons


def test_failing_reload_marker_is_a_loss_location_where_the_slot_may_be_empty():
    c = stim.Circuit("R 0 1\nI_ERROR[reload](0.5) 0 1\nCX 0 1\nI_ERROR[reload](0.5) 0 1\nI_ERROR[reload] 0\nM 0 1")
    assert loss_locations(c, LossModel()) == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_reload_anywhere_sees_a_frame_halfway_through_a_gate_run():
    # H S S H (= X) twice is the identity, but a |0> reloaded after the first run reads 1 at the CX
    run = "H 0\nTICK\nS 0\nTICK\nS 0\nTICK\nH 0\nTICK\n"
    rel = frame_css(stim.Circuit("R 0 1\nRX 2\nCX 2 0\n" + run + run + "CX 0 1\nM 0 1\nMX 2"))
    assert not rel.proven and "reload of qubit 0" in rel.reasons[0]


@pytest.mark.parametrize("marker", ["I_ERROR[reload] 0\n", "I_ERROR[reload](0.5) 0\n"])
def test_loss_families_sample_reload_markers_exactly(marker):
    c = stim.Circuit("R 0 1\nRX 2\nCX 2 0\n" + marker + "CX 0 1\nM 0 1\nMX 2")
    rel = frame_css(c)
    assert rel.proven
    model = LossModel(after="any")
    always = stim.Circuit("R 0 1\nRX 2\nCX 2 0\nI_ERROR[reload] 0\nCX 0 1\nM 0 1\nMX 2")  # family patterns fix each reload's outcome
    fam = loss_family(c, slots=loss_locations(c, model), n_random=40, max_losses=3, seed=3)
    k_marker = next(k for k, inst in enumerate(c) if mf.is_reload(inst))
    assert any(k_marker in s for s in fam) == ("(" in marker)     # failures are tested where reloads can fail
    for i, s in enumerate(fam):
        n = c.num_measurements + 100
        a, _ = mf.sample(always, n, seed=i, forced_losses=s, reference=rel.reference)
        b = mf.explicit_lossy_circuit(always, s).compile_sampler(seed=i).sample(n)
        assert same_distribution(a, b), s


@pytest.mark.parametrize("text", [
    "R 0 1\nM 0\nCX rec[-1] 1\nM 1",
    "R 0 1\nCX sweep[0] 1\nM 1",
    "R 0 1\nMPP X0*X1\nM 0 1",
    "R 0 1\nMZZ 0 1\nM 0 1",
    "R 0 1\nSPP X0*Z1\nM 0 1",
    "R 0\nHERALDED_ERASE(0.1) 0\nM 0",
])
def test_operations_outside_assumption_i_stop_early(text):
    c = stim.Circuit(text)
    assert frame_css(c).verdict == "outside-assumptions"
    assert certify(c, PESSIMISTIC).verdict == "outside-assumptions"


def test_pauli_noise_is_inside_the_assumptions():
    c = stim.Circuit("R 0 1\nPAULI_CHANNEL_1(0.1, 0, 0) 0\nE(0.1) X0 Z1\nCX 0 1\nM 0 1")
    assert frame_css(c).verdict == "frame-css"


def test_long_swap_chain_neither_recurses_nor_slows_down():
    # each SWAP ties the next segment's frame to the previous one: a union-find chain 3000 deep
    n = 3000
    c = stim.Circuit(f"R {' '.join(map(str, range(n)))}")
    for q in range(n - 1):
        c.append("TICK")
        c.append("SWAP", [q + 1, q])
    c.append("M", list(range(n)))
    rel = frame_css(c)
    assert rel.verdict == "frame-css" and not rel.reference.any()


@pytest.mark.parametrize("reset", ["MR", "MRX", "MRY"])
def test_constraints_read_fused_measure_resets_in_order(reset):
    # Stim fuses two measure-resets of one qubit into `MR 0 0`; the second readout is the deterministic one
    c = stim.Circuit(f"R 0 1\nH 0\nCX 0 1\n{reset} 0\n{reset} 0")    # half a Bell pair: random in every basis
    assert len(c) == 4, "premise: Stim fused the two measure-resets"
    samples = c.compile_sampler(seed=1).sample(500)
    assert len(set(samples[:, 1])) == 1 and len(set(samples[:, 0])) == 2
    assert [(y, v) for y, v in constraints(c, {})] == [(0b10, 0)]


def test_reloads_anywhere_land_between_fused_gates():
    # SQRT_X 0 0 0 0 is four gates (identity overall); a qubit lost after the second and reloaded in |0> reads 1
    c = stim.Circuit("R 1\nSQRT_X 0 0 0 0\nM 0 1")
    assert len(c) == 3, "premise: Stim fused the four gates into one instruction"
    assert frame_css(c).verdict == "not-frame-css"


def test_fused_gates_keep_a_valid_reference():
    # losses only between instructions, but kappa of the fused circuit also holds when a loss may fall between
    # the two CX gates
    fused = stim.Circuit("R 0 1 2\nH 0\nCX 0 1 1 2\nH 0\nM 0 1 2")
    split = stim.Circuit("R 0 1 2\nH 0\nCX 0 1\nTICK\nCX 1 2\nH 0\nM 0 1 2")
    assert len(fused) == 5, "premise: the dependent CX gates share one instruction"
    rel = frame_css(fused)
    assert rel.verdict == frame_css(split).verdict == "frame-css"
    fam = loss_family(split, slots=loss_locations(split, PESSIMISTIC), pairs=True)
    assert check_reference(split, rel.reference, fam) == []
