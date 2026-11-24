"""C++ masked-frame sampler (ellis._core): exact against Stim under gate-drop loss for every Clifford gate (with and
without reload markers at the resets, every SIMD width, several batches per call, repeated targets, inverted
measurements and MPAD), consistent with Stim when nothing is lost, with Stim's m2d of its own measurements, and with
itself across packing and thread counts; unsupported operations are rejected loudly."""
from pathlib import Path

import frame_oracle as fo
import numpy as np
import pytest
import stim
from ellis.check_exact import mark_losses, same_distribution
from ellis.noise import surface_memory

from ellis import frame as mf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

native = pytest.importorskip("ellis._core")


def lossy_surface(d=3, p=0.01, p_loss=0.02):
    return surface_memory(d, d, p, p_loss)


def close(a, b, n, sigmas=5):
    """Bernoulli rates a, b (arrays) estimated from n samples each agree within `sigmas` standard errors."""
    se = np.sqrt((a * (1 - a) + b * (1 - b)) / n) + 1 / n
    return bool((np.abs(a - b) < sigmas * se).all())


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


def random_losses(rng, c, lo, hi):
    """A few losses right after random instructions of c, each on one of the instruction's qubits."""
    flat, losses = list(c), {}
    for _ in range(int(rng.integers(lo, hi))):
        k = int(rng.integers(len(flat)))
        qs = [x.value for x in flat[k].targets_copy() if x.is_qubit_target]
        if qs:
            losses.setdefault(k, []).append(int(rng.choice(qs)))
    return losses


def native_matches_stim(c, losses, seed, simd_width, batch_size):
    """The masked frame, with this loss pattern's own exact reference, against Stim on the explicit lossy circuit
    (so only the frame propagation is tested, not the choice of reference)."""
    ex = mf.explicit_lossy_circuit(c, losses)
    ref = mf.strip_loss(ex).reference_sample()
    n = c.num_measurements + 64
    s = native.MaskFrameSampler(mark_losses(c, losses), reference=ref, seed=seed, simd_width=simd_width, batch_size=batch_size)
    return same_distribution(s.sample(n)[0], ex.compile_sampler(seed=seed).sample(n))


@pytest.mark.parametrize("simd_width,batch_size", [(256, 256), (128, 128), (64, 64), (0, 0)])
def test_every_clifford_gate_matches_stim_under_loss(simd_width, batch_size):
    rng = np.random.default_rng(11)
    for t in range(150):
        c = random_circuit(rng)
        if t % 2:
            c, _ = fo.reload_at_resets(c)
        losses = random_losses(rng, c, 0, 3)
        assert native_matches_stim(c, losses, t, simd_width, batch_size), (str(c), losses)


def test_repeated_targets_inversions_and_mpad():
    assert native.MaskFrameSampler(stim.Circuit("RX 0\nS 0 0\nMX 0"), seed=1).sample(200)[0].all()
    c = stim.Circuit("R 0\nM !0\nMPAD 1 0\nM 0")
    assert (native.MaskFrameSampler(c, seed=1).sample(50)[0] == [1, 1, 0, 0]).all()
    assert mf.explicit_lossy_circuit(c, {}).reference_sample().tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("text", ["M 0\nCX rec[-1] 1", "MPP X0*X1", "HERALDED_ERASE(0.1) 0",
                                  "I_ERROR[reload](0.5, 0.5) 0\nM 0", "R 0\nM 0\nOBSERVABLE_INCLUDE(0) X0"])
def test_rejects_unsupported(text):
    with pytest.raises(ValueError):
        native.MaskFrameSampler(stim.Circuit(text))
    if "OBSERVABLE_INCLUDE" not in text:
        with pytest.raises(ValueError):
            mf.validate(stim.Circuit(text))


def test_loss_drops_gates_and_heralds():
    plain = stim.Circuit("R 0 1\nX 0\nCX 0 1\nM 0 1\nR 0\nM 0")
    for reload in (False, True):
        c, move = fo.reload_at_resets(plain) if reload else (plain, lambda s: s)
        meas, _, _, her = native.MaskFrameSampler(mark_losses(c, move({1: [0]})), seed=2).sample(4000)
        assert her[:, 0].all() and not her[:, 1].any() and (her[:, 2] == (not reload)).all()
        assert not meas[:, 1].any()                       # the CX was dropped
        assert 0.45 < meas[:, 0].mean() < 0.55            # lost readout: uniform placeholder
        assert reload == (not meas[:, 2].any())           # a restored qubit reads 0


@pytest.mark.parametrize("p", [0.001, 0.3, 0.9])
def test_loss_probability_per_site(p):
    """Each I_ERROR[loss](p) target is lost independently with probability p (Poisson hits are exact)."""
    k, n = 3, 200000
    c = stim.Circuit("R 0 1 2 3\n" + f"I_ERROR[loss]({p}) 0 1 2 3\n" * k + "M 0 1 2 3")
    her = native.MaskFrameSampler(c, seed=4, batch_size=1024).sample(n)[3]
    lost = 1 - (1 - p) ** k
    assert close(her.mean(0), np.full(4, lost), n)
    assert close(np.array([(her[:, 0] & her[:, 1]).mean()]), np.array([lost ** 2]), n)   # independent sites


def test_default_reference_is_pythons():
    for path in sorted((FIXTURES / "lightstim").glob("*.stim")):
        c = stim.Circuit(path.read_text())
        assert (native.MaskFrameSampler(c).reference == mf.strip_loss(c).reference_sample()).all(), path.name


def surface_with_reloads(reload, **kw):
    c = lossy_surface(**kw)
    return fo.reload_at_resets(c)[0] if reload else c


@pytest.mark.parametrize("reload", [False, True])
def test_detectors_are_measurement_parities(reload):
    """The detectors and observables computed in the simulation pass are Stim's m2d of its measurements."""
    c = surface_with_reloads(reload, p_loss=0.03)
    meas, dets, obs, her = native.MaskFrameSampler(c, seed=5, batch_size=512).sample(3000)
    d2, o2 = c.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    assert (dets == d2).all() and (obs == o2).all() and her.any() and dets.any()


def test_pauli_gates_and_custom_reference_keep_stims_detector_convention():
    c = stim.Circuit("R 0 1\nX 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    for ref in (None, [0, 1]):          # [0, 1] flips measurement 1: a detection event in every shot
        s = native.MaskFrameSampler(c, reference=ref, seed=1)
        meas, dets, obs, _ = s.sample(100)
        assert (meas == [1, ref is not None]).all() and not dets[:, 0].any() and not obs.any()
        assert (dets[:, 1] == (ref is not None)).all()


def test_threads_and_bit_packing_do_not_change_results():
    c = lossy_surface()
    def run(**kw):
        return native.MaskFrameSampler(c, seed=3, batch_size=256).sample(5000, **kw)
    base, multi, packed = run(), run(threads=4), run(bit_packed=True)
    assert all((x == y).all() for x, y in zip(base, multi))
    assert all((np.packbits(x, axis=1, bitorder="little") == y).all() for x, y in zip(base, packed))


def test_bit_identical_to_stim_frame_simulator_when_nothing_is_lost():
    """Built on stim::FrameSimulator with Stim's random stream: with loss probability 0 the samples are Stim's.
    Not with reload markers: a marker draws its fresh gauge bits in every shot, lost or not."""
    c = lossy_surface(p=0.01, p_loss=0.0)
    def pair(simd_width):
        kw = {"seed": 7, "batch_size": 512, "simd_width": simd_width}
        return native.MaskFrameSampler(c, **kw), native.StimSampler(c, **kw)
    for simd_width in native.SIMD_WIDTHS:
        a, b = pair(simd_width)
        assert all((x == y).all() for x, y in zip(a.sample(2000), b.sample(2000)))


def test_lossless_detector_rates_match_stim():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                               after_clifford_depolarization=0.02, before_round_data_depolarization=0.02,
                               before_measure_flip_probability=0.02, after_reset_flip_probability=0.02)
    n = 50000
    d0 = np.asarray(c.compile_detector_sampler(seed=1).sample(n)).mean(0)
    d1 = native.MaskFrameSampler(c, seed=1).sample(n)[1].mean(0)
    d2 = native.StimSampler(c, seed=1).sample(n)[1].mean(0)
    assert close(d0, d1, n) and close(d0, d2, n)


@pytest.mark.parametrize("p", [0.004, 0.3])      # sparse (geometric gaps) and dense (biased rows) noise paths
def test_noise_channels_match_stim(p):
    noise = (f"X_ERROR({p}) 0\nY_ERROR({p}) 1\nZ_ERROR({p}) 2\nDEPOLARIZE1({p}) 3\nDEPOLARIZE2({p}) 4 5\n"
             f"PAULI_CHANNEL_1({p / 2}, {p / 3}, {p / 4}) 6\nPAULI_CHANNEL_2({', '.join(str(p * (k + 1) / 120) for k in range(15))}) 7 8")
    n = 100000
    for prep, meas in (("R", "M"), ("RX", "MX")):
        c = stim.Circuit(f"{prep} 0 1 2 3 4 5 6 7 8\n{noise}\n{meas}({p / 3}) 0 1 2 3 4 5 6 7 8")
        def stats(m):
            return np.concatenate([m.mean(0), [(m[:, 4] & m[:, 5]).mean(), (m[:, 7] & m[:, 8]).mean()]])
        a = stats(c.compile_sampler(seed=1).sample(n))
        b = stats(native.MaskFrameSampler(c, seed=1).sample(n)[0])
        assert close(a, b, n), (prep, a, b)


@pytest.mark.parametrize("simd_width,batch_size", [(256, 256), (64, 64)])
def test_reload_markers_match_stim_under_loss(simd_width, batch_size):
    rng = np.random.default_rng(29)
    for t in range(100):
        c = random_circuit(rng, reload_markers=True)
        if t % 2:
            c, _ = fo.reload_at_resets(c)
        losses = random_losses(rng, c, 1, 4)
        assert native_matches_stim(c, losses, t, simd_width, batch_size), (str(c), losses)


def test_reload_marker_restores_a_lost_atom_in_zero():
    c = stim.Circuit("R 0 1\nX 0\nI_ERROR[loss](1) 0\nI_ERROR[reload] 0\nM 0\nCX 0 1\nM 1")
    m, _, _, h = native.MaskFrameSampler(c, seed=3).sample(200)
    assert not m.any() and not h.any()
    m = native.MaskFrameSampler(stim.Circuit("RX 0\nI_ERROR[reload] 0\nMX 0"), seed=3).sample(200)[0]
    assert not m.any()                                    # a present atom keeps |+>


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_reload_marker_succeeds_with_probability_p(p):
    n = 20000
    m, _, _, h = native.MaskFrameSampler(stim.Circuit(f"R 0\nI_ERROR[loss](1) 0\nI_ERROR[reload]({p}) 0\nM 0"), seed=4).sample(n)
    assert abs(h.mean() - (1 - p)) < 5 * np.sqrt(p * (1 - p) / n) + 1e-9
    assert not m[~h].any()                                # a reloaded atom reads its fresh |0>
