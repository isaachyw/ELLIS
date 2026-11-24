"""ellis.compile_sampler: routing by certificate, result layout on both routes, and the C++ masked frame and Clifft
sampling the same distributions (exactly with deterministic loss, statistically with random loss and noise)."""
import sys

import frame_oracle as fo
import numpy as np
import pytest
import stim
from ellis.certify import Certificate, LossModel, certify
from ellis.check_exact import mark_losses, same_distribution
from ellis.noise import surface_memory
from ellis.sampler import route
from test_frame import close, random_circuit, random_losses

import ellis
from ellis import _clifft
from ellis import frame as mf

pytest.importorskip("clifft")

METHODS = ["frame", "clifft"]
# Contextual: no single reference fits every loss pattern of its markers (test_certify's contradiction circuit).
CONTEXTUAL = stim.Circuit("RX 2\nCX 2 0\nI_ERROR[loss](0.1) 2 0\nCZ 0 2\nI_ERROR[loss](0.1) 0 2\nCX 2 0\n"
                          "I_ERROR[loss](0.1) 2 0\nMX 1\nCX 1 2\nI_ERROR[loss](0.1) 1 2\nMX 1")


def test_routes_by_certificate():
    s = ellis.compile_sampler(surface_memory(3, 3, 1e-3, 1e-2), seed=1)
    assert (s.method, s.certificate.verdict, s.certificate.method) == ("frame", "proven", "frame-css")
    s = ellis.compile_sampler(CONTEXTUAL, seed=1)
    assert (s.method, s.certificate.verdict) == ("clifft", "contextual") and "contextual" in s.reason
    # proven only with a gauge, which the C++ frame does not apply: Clifft samples it exactly (qubit 0 ends in |->)
    gauged = stim.Circuit("RX 0 1 2\nCZ 0 1 1 2 0 2\nCZ 0 1 1 2 0 2\nS 0\nS 0\nMX 0 1 2")
    cert = certify(gauged, LossModel(after="any", noiseless_loss_free=False, from_start=True))
    s = ellis.compile_sampler(gauged, certificate=cert, seed=1)
    assert cert.method == "diagonal-blocks+gauge" and s.method == "clifft"
    assert (s.sample(100).measurements == [1, 0, 0]).all()
    for verdict in ("contextual", "consistent_on_family", "outside-assumptions"):
        assert route(Certificate(verdict, "exact-search", np.zeros(3, bool)))[0] == "clifft"
    assert route(Certificate("proven", "frame-css", np.zeros(3, bool)))[0] == "frame"


def test_explicit_methods_skip_certify():
    for method in METHODS:
        s = ellis.compile_sampler(CONTEXTUAL, method=method, seed=1)
        assert s.method == method and s.certificate is None
    with pytest.raises(ValueError):
        ellis.compile_sampler(CONTEXTUAL, method="numpy")


@pytest.mark.parametrize("method", ["auto", *METHODS])
@pytest.mark.parametrize("text", ["R 0 1\nMPP X0*X1", "M 0\nCX rec[-1] 1", "R 0\nHERALDED_ERASE(0.1) 0\nM 0"])
def test_rejects_non_clifford_operations(method, text):
    with pytest.raises(ValueError):
        ellis.compile_sampler(stim.Circuit(text), method=method)


def test_missing_clifft_names_the_verdict(monkeypatch):
    monkeypatch.setitem(sys.modules, "clifft", None)
    with pytest.raises(ImportError, match="contextual.*ellis\\[clifft\\]"):
        ellis.compile_sampler(CONTEXTUAL)
    assert ellis.compile_sampler(surface_memory(3, 3, 1e-3, 1e-2)).method == "frame"


def test_auto_certifies_only_the_marked_losses():
    # contextual under PESSIMISTIC (certify's default without markers), but sampling never loses an atom here
    unmarked = stim.Circuit("RX 2\nCX 2 0\nCZ 0 2\nCX 2 0\nMX 1\nCX 1 2\nMX 1")
    assert certify(unmarked, n_random=0).verdict == "contextual"
    s = ellis.compile_sampler(unmarked, seed=1)
    assert (s.method, s.certificate.verdict) == ("frame", "proven")
    assert ellis.compile_sampler(unmarked, certify_options={"model": certify(unmarked).model}).method == "clifft"


@pytest.mark.parametrize("method", METHODS)
def test_result_layout_and_bit_packing(method):
    c = surface_memory(3, 3, 0.01, 0.02)
    n, M, D, O = 300, c.num_measurements, c.num_detectors, c.num_observables
    s = ellis.compile_sampler(c, method=method, seed=5)
    assert (s.num_qubits, s.num_measurements, s.num_detectors, s.num_observables) == (c.num_qubits, M, D, O)
    r = s.sample(n)
    assert [x.shape for x in (r.measurements, r.detectors, r.observables, r.heralds)] == [(n, M), (n, D), (n, O), (n, M)]
    assert all(x.dtype == bool for x in (r.measurements, r.detectors, r.observables, r.heralds))
    d, o = c.compile_m2d_converter().convert(measurements=r.measurements, separate_observables=True)
    assert (d == r.detectors).all() and (o == r.observables).all() and r.heralds.any()
    packed = ellis.compile_sampler(c, method=method, seed=5).sample(n, bit_packed=True)
    for name in ("measurements", "detectors", "observables", "heralds"):
        a, b = getattr(r, name), getattr(packed, name)
        assert b.dtype == np.uint8 and (np.packbits(a, axis=1, bitorder="little") == b).all(), name
    r = s.sample_detectors(n)
    assert r.measurements is None and r.heralds is None and r.detectors.shape == (n, D) and r.detectors.dtype == bool
    r = s.sample_detectors(n, bit_packed=True, heralds=True, threads=2)
    assert [x.shape for x in (r.detectors, r.observables, r.heralds)] == [(n, (D + 7) // 8), (n, 1), (n, (M + 7) // 8)]


@pytest.mark.parametrize("method", METHODS)
def test_noiseless_lossless_detectors_are_zero(method):
    # raw parities are 1 here (X, `M !q`, MPAD 1); detection events are relative to the noiseless reference
    c = stim.Circuit("R 0 1\nX 0\nM 0 !1\nMPAD 1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
                     "OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]") + surface_memory(3, 3, 0.0, 0.0)
    s = ellis.compile_sampler(c, method=method, seed=2)
    for r in (s.sample(200), s.sample_detectors(200, heralds=True)):
        assert not r.detectors.any() and not r.observables.any() and not r.heralds.any()


def frame_and_clifft(c, losses, n):
    """Measurements and heralds of `c` with `losses` (p = 1 markers) from the masked frame, given this loss pattern's
    exact reference, and from Clifft."""
    marked = mark_losses(c, losses)
    exact = Certificate("proven", "frame-css", mf.strip_loss(mf.explicit_lossy_circuit(c, losses)).reference_sample())
    out = []
    for method in METHODS:
        r = ellis.compile_sampler(marked, method=method, certificate=exact, seed=3).sample(n)
        out.append((r.measurements, r.heralds))
    return out


@pytest.mark.parametrize("text,losses", [
    ("RX 0\nR 1\nCX 0 1\nMR 0 1\nCX 0 1\nH 1\nM 0 1", {2: [0]}),                 # MR on a lost qubit keeps it lost
    ("RX 0\nR 1\nCX 0 1\nM 0\nI_ERROR[reload] 0\nRX 0\nCZ 0 1\nMX 0 1", {2: [0]}),  # reload, then entangled again
    ("RX 0 1\nCX 0 1\nMPAD 1 0\nM !0\nMX !1", {1: [1]}),                         # MPAD, inverted readout of a lost qubit
    ("RX 0\nR 1 2\nCX 0 1\nCX 1 2\nCZ 0 2\nH 0\nM 0 1 2", {1: [1]}),             # gates with a lost partner
    ("R 0 1\nX 0\nI_ERROR[reload] 0\nCX 0 1\nM 0 1", {-1: [0]}),                 # reload of an atom lost from the start
    ("RX 0\nI_ERROR[reload] 0\nMX 0", {}),                                        # a present atom keeps its state
])
def test_frame_and_clifft_agree_exactly_on_deterministic_loss(text, losses):
    c = stim.Circuit(text)
    (a, ha), (b, hb) = frame_and_clifft(c, losses, c.num_measurements + 64)
    assert same_distribution(a, b) and (ha == hb).all() and (ha == ha[0]).all()


def test_frame_and_clifft_agree_exactly_on_random_circuits():
    rng = np.random.default_rng(31)
    for t in range(60):
        c = random_circuit(rng, reload_markers=t % 3 == 0)
        if t % 2:
            c, _ = fo.reload_at_resets(c)
        losses = random_losses(rng, c, 1, 4)
        (a, ha), (b, hb) = frame_and_clifft(c, losses, c.num_measurements + 64)
        assert same_distribution(a, b) and (ha == hb).all(), (str(c), losses)


@pytest.mark.parametrize("method", METHODS)
def test_two_qubit_noise_skips_a_present_partner_of_a_lost_atom(method):
    n = 40000
    r = ellis.compile_sampler(stim.Circuit("R 0 1\nI_ERROR[loss](0.5) 0\nDEPOLARIZE2(0.3) 0 1\nM 0 1"),
                              method=method, seed=4).sample(n)
    lost = r.heralds[:, 0]
    assert not r.heralds[:, 1].any() and not r.measurements[lost, 1].any()
    assert close(r.measurements[~lost, 1].mean(), np.array(0.3 * 8 / 15), (~lost).sum())   # X or Y on qubit 1


@pytest.mark.parametrize("text", [
    "R 0 1\nX 0\nI_ERROR[loss](0.5) 0\nI_ERROR[reload] 0\nH 0\nCX 0 1\nM 0 1",
    "R 0 1\nX 0\nI_ERROR[loss](0.5) 0\nI_ERROR[reload](0.4) 0\nH 0\nCX 0 1\nM 0 1",
])
def test_frame_and_clifft_agree_on_random_reloads(text):
    n, c = 20000, stim.Circuit(text)
    (a, ha), (b, hb) = frame_and_clifft(c, {}, n)
    assert close(a.mean(0), b.mean(0), n) and close(ha.mean(0), hb.mean(0), n)
    for m, h in ((a, ha), (b, hb)):
        assert (m[~h[:, 0], 0] == m[~h[:, 0], 1]).all()        # a present or reloaded atom stays correlated


@pytest.mark.parametrize("reload", [False, True])
def test_frame_and_clifft_agree_on_lossy_surface_memory(reload):
    c, n = surface_memory(3, 3, 0.01, 0.03), 20000
    if reload:
        c = fo.reload_at_resets(c)[0]
    s = ellis.compile_sampler(c, seed=6)
    assert s.method == "frame"
    a = s.sample_detectors(n, heralds=True)
    b = ellis.compile_sampler(c, method="clifft", seed=6).sample_detectors(n, heralds=True)
    assert close(a.detectors.mean(0), b.detectors.mean(0), n) and close(a.heralds.mean(0), b.heralds.mean(0), n)
    assert close(a.observables.mean(0), b.observables.mean(0), n) and 0.1 < a.heralds.any(1).mean() < 0.99


def test_clifft_online_active_width_stays_small_on_lossy_clifford_circuits():
    program = _clifft.Backend(surface_memory(5, 5, 1e-3, 1e-2), seed=1).program
    stats = program.sample(2000, seed=1, collect_stats=True).stats
    assert stats.strategy == "online" and stats.peak_active_width <= 1
