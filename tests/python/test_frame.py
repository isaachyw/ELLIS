"""Masked-frame sampler: exact against Stim under gate-drop loss (both reload modes), for every Stim Clifford
gate, repeated targets, inverted measurements and MPAD; unsupported operations are rejected loudly."""
import numpy as np
import pytest
import stim
from ellis.check_exact import same_distribution

from ellis import frame as mf


def matches_stim(c, losses, reload, seed):
    """The masked frame, with this loss pattern's own exact reference, against Stim on the explicit
    lossy circuit (so only the frame propagation is tested, not the choice of reference)."""
    ex = mf.explicit_lossy_circuit(c, losses, reload)
    ref = mf.strip_loss(ex).reference_sample()
    n = c.num_measurements + 64
    a, _ = mf.sample(c, n, seed=seed, reload=reload, forced_losses=losses, reference=ref)
    return same_distribution(a, ex.compile_sampler(seed=seed).sample(n))


def random_circuit(rng, n=4, depth=14, reload_markers=False):
    one, two = sorted(mf.UNITARY_1Q), sorted(mf.UNITARY_2Q)
    c = stim.Circuit()
    for _ in range(depth):
        if reload_markers and rng.random() < 0.2:
            c.append("I_ERROR", [int(rng.integers(n))], tag="reload")
        r = rng.random()
        if r < 0.3:                                   # repeated targets allowed, e.g. `S 1 1`
            c.append(str(rng.choice(one)), [int(q) for q in rng.integers(0, n, rng.integers(1, 4))])
        elif r < 0.6:
            a, b = rng.choice(n, 2, replace=False)
            c.append(str(rng.choice(two)), [int(a), int(b)])
        elif r < 0.7:
            c.append(str(rng.choice(["R", "RX", "RY"])), [int(rng.integers(n))])
        elif r < 0.95:
            q = int(rng.integers(n))
            c.append(str(rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY"])), [stim.target_inv(q) if rng.random() < 0.3 else q])
        else:
            c.append("MPAD", [int(rng.integers(2))])
    c.append("M", list(range(n)))
    return c


def test_repeated_targets_act_twice():
    m, _ = mf.sample(stim.Circuit("RX 0\nS 0 0\nMX 0"), 200, seed=1)       # S.S = Z maps |+> to |->
    assert m.all()


def test_inverted_measurement_and_mpad():
    c = stim.Circuit("R 0\nM !0\nMPAD 1 0\nM 0")
    m, _ = mf.sample(c, 50, seed=1)
    assert (m == [1, 1, 0, 0]).all()
    assert mf.explicit_lossy_circuit(c, {}).reference_sample().tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("text", ["M 0\nCX rec[-1] 1", "MPP X0*X1", "HERALDED_ERASE(0.1) 0",
                                  "I_ERROR[reload](0.5) 0\nM 0"])
def test_rejects_unsupported(text):
    with pytest.raises(ValueError):
        mf.sample(stim.Circuit(text), 10)


def test_every_clifford_gate_matches_stim_under_loss():
    rng = np.random.default_rng(7)
    for t in range(200):
        c = random_circuit(rng)
        flat, losses = list(c), {}
        for _ in range(int(rng.integers(0, 3))):
            k = int(rng.integers(len(flat)))
            qs = [x.value for x in flat[k].targets_copy() if x.is_qubit_target]
            if qs:
                losses.setdefault(k, []).append(int(rng.choice(qs)))
        assert matches_stim(c, losses, bool(t % 2), t), (str(c), losses)


def test_reload_markers_match_stim_under_loss():
    # `I_ERROR[reload] q` anywhere: a fresh |0> in an empty slot, a present atom untouched
    rng = np.random.default_rng(23)
    for t in range(150):
        c = random_circuit(rng, reload_markers=True)
        flat, losses = list(c), {}
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(len(flat)))
            qs = [x.value for x in flat[k].targets_copy() if x.is_qubit_target]
            if qs:
                losses.setdefault(k, []).append(int(rng.choice(qs)))
        assert matches_stim(c, losses, bool(t % 2), t), (str(c), losses)


def test_reload_marker_restores_a_lost_atom_in_zero():
    c = stim.Circuit("R 0 1\nX 0\nI_ERROR[loss](1) 0\nI_ERROR[reload] 0\nM 0\nCX 0 1\nM 1")
    m, h = mf.sample(c, 200, seed=3)
    assert not m.any() and not h.any()
    m, _ = mf.sample(stim.Circuit("RX 0\nI_ERROR[reload] 0\nMX 0"), 200, seed=3)   # a present atom keeps |+>
    assert not m.any()
