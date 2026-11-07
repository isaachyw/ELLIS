"""C++ masked-frame sampler (ellis._core): exact against Stim under gate-drop loss for every Clifford gate
(both reload modes, every SIMD width, several batches per call), and consistent with the numpy sampler, with Stim
when nothing is lost, and with itself across output modes, packing and thread counts."""
from pathlib import Path

import numpy as np
import pytest
import stim
from ellis.check_exact import same_distribution
from ellis.noise import surface_memory
from test_frame import random_circuit

from ellis import frame as mf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

native = pytest.importorskip("ellis._core")


def lossy_surface(d=3, p=0.01, p_loss=0.02):
    return surface_memory(d, d, p, p_loss)


def close(a, b, n, sigmas=5):
    """Bernoulli rates a, b (arrays) estimated from n samples each agree within `sigmas` standard errors."""
    se = np.sqrt((a * (1 - a) + b * (1 - b)) / n) + 1 / n
    return bool((np.abs(a - b) < sigmas * se).all())


def native_matches_stim(c, losses, reload, seed, simd_width, batch_size):
    """As test_frame.matches_stim: the loss pattern's own exact reference, so only the frame is tested."""
    ex = mf.explicit_lossy_circuit(c, losses, reload)
    ref = mf.strip_loss(ex).reference_sample()
    n = c.num_measurements + 64
    s = mf.compile_sampler(c, reload=reload, forced_losses=losses, reference=ref, seed=seed,
                           simd_width=simd_width, batch_size=batch_size)
    a, _ = s.sample(n)
    return same_distribution(a, ex.compile_sampler(seed=seed).sample(n))


@pytest.mark.parametrize("simd_width,batch_size", [(256, 256), (128, 128), (64, 64), (0, 0)])
def test_every_clifford_gate_matches_stim_under_loss(simd_width, batch_size):
    rng = np.random.default_rng(11)
    for t in range(150):
        c = random_circuit(rng)
        flat, losses = list(c), {}
        for _ in range(int(rng.integers(0, 3))):
            k = int(rng.integers(len(flat)))
            qs = [x.value for x in flat[k].targets_copy() if x.is_qubit_target]
            if qs:
                losses.setdefault(k, []).append(int(rng.choice(qs)))
        assert native_matches_stim(c, losses, bool(t % 2), t, simd_width, batch_size), (str(c), losses)


def test_repeated_targets_inversions_and_mpad():
    assert native.MaskFrameSampler(stim.Circuit("RX 0\nS 0 0\nMX 0"), seed=1).sample(200, heralds=False).all()
    c = stim.Circuit("R 0\nM !0\nMPAD 1 0\nM 0")
    assert (native.MaskFrameSampler(c, seed=1).sample(50, heralds=False) == [1, 1, 0, 0]).all()


@pytest.mark.parametrize("text", ["M 0\nCX rec[-1] 1", "MPP X0*X1", "HERALDED_ERASE(0.1) 0",
                                  "R 0\nM 0\nOBSERVABLE_INCLUDE(0) X0"])
def test_rejects_unsupported(text):
    with pytest.raises(ValueError):
        native.MaskFrameSampler(stim.Circuit(text))


def test_forced_loss_drops_gates_and_heralds():
    c = stim.Circuit("R 0 1\nX 0\nCX 0 1\nM 0 1\nR 0\nM 0")
    for reload in (False, True):
        meas, her = native.MaskFrameSampler(c, reload=reload, forced_losses={1: [0]}, seed=2).sample(4000)
        assert her[:, 0].all() and not her[:, 1].any() and (her[:, 2] == (not reload)).all()
        assert not meas[:, 1].any()                       # the CX was dropped
        assert 0.45 < meas[:, 0].mean() < 0.55            # lost readout: uniform placeholder
        assert reload == (not meas[:, 2].any())           # a restored qubit reads 0


@pytest.mark.parametrize("p", [0.001, 0.3, 0.9])
def test_loss_probability_per_site(p):
    """Each I_ERROR[loss](p) target is lost independently with probability p (Poisson hits are exact)."""
    k, n = 3, 200000
    c = stim.Circuit("R 0 1 2 3\n" + f"I_ERROR[loss]({p}) 0 1 2 3\n" * k + "M 0 1 2 3")
    _, her = native.MaskFrameSampler(c, seed=4, batch_size=1024).sample(n)
    lost = 1 - (1 - p) ** k
    assert close(her.mean(0), np.full(4, lost), n)
    assert close(np.array([(her[:, 0] & her[:, 1]).mean()]), np.array([lost ** 2]), n)   # independent sites


def test_default_reference_is_pythons():
    for path in sorted((FIXTURES / "lightstim").glob("*.stim")):
        c = stim.Circuit(path.read_text())
        assert (native.MaskFrameSampler(c).reference == mf.strip_loss(c).reference_sample()).all(), path.name


@pytest.mark.parametrize("reload", [False, True])
def test_detectors_are_measurement_parities(reload):
    """Same seed and batch size: detector mode (ring-buffered record) reproduces Stim's m2d of sample()."""
    c = lossy_surface(p_loss=0.03)
    meas, her = native.MaskFrameSampler(c, reload=reload, seed=5, batch_size=512).sample(3000)
    dets, obs, her2 = native.MaskFrameSampler(c, reload=reload, seed=5, batch_size=512).sample_detectors(3000, heralds=True)
    d2, o2 = c.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    assert (dets == d2).all() and (obs == o2).all() and (her == her2).all()


def test_pauli_gates_and_custom_reference_keep_stims_detector_convention():
    c = stim.Circuit("R 0 1\nX 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    for ref in (None, [0, 1]):          # [0, 1] flips measurement 1: a detection event in every shot
        s = native.MaskFrameSampler(c, reference=ref, seed=1)
        meas = s.sample(100, heralds=False)
        dets, obs = s.sample_detectors(100)
        assert (meas == [1, ref is not None]).all() and not dets[:, 0].any() and not obs.any()
        assert (dets[:, 1] == (ref is not None)).all()


def test_threads_and_bit_packing_do_not_change_results():
    c = lossy_surface()
    def run(**kw):
        return native.MaskFrameSampler(c, seed=3, batch_size=256).sample(5000, **kw)
    base, multi, packed = run(), run(threads=4), run(bit_packed=True)
    assert all((x == y).all() for x, y in zip(base, multi))
    assert all((np.packbits(x, axis=1, bitorder="little") == y).all() for x, y in zip(base, packed))
    dets = native.MaskFrameSampler(c, seed=3).sample_detectors(5000, threads=3)
    dets_packed = native.MaskFrameSampler(c, seed=3).sample_detectors(5000, bit_packed=True)
    assert all((np.packbits(x, axis=1, bitorder="little") == y).all() for x, y in zip(dets, dets_packed))


@pytest.mark.parametrize("reload", [False, True])
def test_agrees_with_numpy_sampler(reload):
    c, n = lossy_surface(p_loss=0.03), 20000
    conv = c.compile_m2d_converter()
    m1, h1 = mf.sample(c, n, seed=1, reload=reload)
    m2, h2 = native.MaskFrameSampler(c, reload=reload, seed=1).sample(n)
    d1, d2 = (conv.convert(measurements=m, separate_observables=True)[0] for m in (m1, m2))
    assert close(h1.mean(0), h2.mean(0), n) and close(d1.mean(0), d2.mean(0), n)
    assert 0.1 < h2.any(1).mean() < 0.99


@pytest.mark.parametrize("reload", [False, True])
def test_bit_identical_to_stim_frame_simulator_when_nothing_is_lost(reload):
    """Built on stim::FrameSimulator with Stim's random stream: with loss probability 0 the samples are Stim's."""
    c = lossy_surface(p=0.01, p_loss=0.0)
    def pair(simd_width):
        kw = {"seed": 7, "batch_size": 512, "simd_width": simd_width}
        return native.MaskFrameSampler(c, reload=reload, **kw), native.StimSampler(c, **kw)
    for simd_width in native.SIMD_WIDTHS:
        a, b = pair(simd_width)
        assert (a.sample(2000, heralds=False) == b.sample(2000)).all()
        a, b = pair(simd_width)
        assert all((x == y).all() for x, y in zip(a.sample_detectors(2000), b.sample_detectors(2000)))


def test_lossless_detector_rates_match_stim():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                               after_clifford_depolarization=0.02, before_round_data_depolarization=0.02,
                               before_measure_flip_probability=0.02, after_reset_flip_probability=0.02)
    n = 50000
    d0 = np.asarray(c.compile_detector_sampler(seed=1).sample(n)).mean(0)
    d1 = native.MaskFrameSampler(c, seed=1).sample_detectors(n)[0].mean(0)
    d2 = native.StimSampler(c, seed=1).sample_detectors(n)[0].mean(0)
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
        b = stats(native.MaskFrameSampler(c, seed=1).sample(n, heralds=False))
        assert close(a, b, n), (prep, a, b)


@pytest.mark.parametrize("simd_width,batch_size", [(256, 256), (64, 64)])
def test_reload_markers_match_stim_under_loss(simd_width, batch_size):
    rng = np.random.default_rng(29)
    for t in range(100):
        c = random_circuit(rng, reload_markers=True)
        flat, losses = list(c), {}
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(len(flat)))
            qs = [x.value for x in flat[k].targets_copy() if x.is_qubit_target]
            if qs:
                losses.setdefault(k, []).append(int(rng.choice(qs)))
        assert native_matches_stim(c, losses, bool(t % 2), t, simd_width, batch_size), (str(c), losses)


def test_reload_marker_agrees_with_numpy_sampler():
    c = stim.Circuit("R 0 1\nX 0\nI_ERROR[loss](0.5) 0\nI_ERROR[reload] 0\nH 0\nCX 0 1\nM 0 1")
    n = 20000
    m1, h1 = mf.sample(c, n, seed=1)
    m2, h2 = native.MaskFrameSampler(c, seed=1).sample(n)
    assert close(m1.mean(0), m2.mean(0), n) and not h1.any() and not h2.any()
    assert (m2[:, 0] == m2[:, 1]).all()
