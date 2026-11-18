"""frame_css against independent oracles (frame_oracle.py) under the pessimistic loss model it proves for: an
accepted reference is common to every loss pattern of every loss model (P1), circuits that are frame-CSS by
construction are accepted (P2), the verdict equals a brute-force frame search (P3), operations outside assumption
(i) are "outside-assumptions" (P4), verdict-preserving rewrites keep the verdict (P5), and a contradiction proven
under any loss model excludes frame-CSS (P6)."""
import numpy as np
import pytest
import stim
from ellis.certify import PESSIMISTIC, LossModel, certify, frame_css, loss_family, loss_locations, search

import frame_oracle as fo

MODELS = {"pessimistic": PESSIMISTIC}
WEAKER = {
    "2q": LossModel(),
    "2q-start": LossModel(from_start=True),
    "any": LossModel(after="any"),
    "any-tags-lossy": LossModel(after="any", noiseless_loss_free=False),
    "every": LossModel(after="any", noiseless_loss_free=False, from_start=True),
    "reset": LossModel(after="any", reload="reset"),
    "anywhere": LossModel(after="any", reload="anywhere"),
    "anywhere-2q": LossModel(reload="anywhere"),
    "mixed": LossModel(after="any", reload_state="mixed"),
    "mixed-anywhere": LossModel(after="any", reload="anywhere", reload_state="mixed"),
}


def seed(*keys) -> np.random.Generator:
    return np.random.default_rng([sum(map(ord, str(k))) for k in keys])


def oracle(c: stim.Circuit, model: LossModel) -> str:
    return "frame-css" if fo.brute_force_frame_css(c, model)[0] else "not-frame-css"


def constructive(rng, model: LossModel, per_set: int):
    for gate_set in fo.GATE_SETS:
        for _ in range(per_set):
            n = int(rng.integers(2, 5))
            yield f"{gate_set}/resets", fo.random_frame_css(rng, n, int(rng.integers(4, 12)), gate_set, model, "resets")


def corpus(rng, model: LossModel, n_random: int, per_set: int):
    """(source, circuit): random Clifford circuits, constructive ones, and single-edit near misses of those."""
    for _ in range(n_random):
        yield "random", fo.random_clifford(rng, int(rng.integers(1, 5)), int(rng.integers(3, 12)))
        yield "random-css-able", fo.random_clifford(rng, int(rng.integers(2, 5)), int(rng.integers(3, 12)), fo.CSS_ABLE_2Q)
    for src, c in constructive(rng, model, per_set):
        yield src, c
        for kind, m in fo.mutations(c, rng, model).items():
            yield f"{src}/{kind}", m


def report(bad) -> str:
    bad = sorted(bad, key=lambda b: len(str(b[-1])))
    return f"{len(bad)} cases, smallest first:\n" + "\n---\n".join(" ".join(map(str, b[:-1])) + f"\n{b[-1]}" for b in bad[:3])


# ------------------------------------------------------------------ P1 soundness
@pytest.mark.parametrize("name", MODELS)
def test_p1_accepted_reference_is_a_possible_record_of_every_lossy_circuit(name):
    model, rng = MODELS[name], seed("p1", name)
    bad, accepted = [], 0
    for src, c in corpus(rng, model, n_random=12, per_set=4):
        r = frame_css(c)
        if r.verdict != "frame-css":
            continue
        accepted += 1
        if model.reload_state == "zero":
            patterns = fo.loss_patterns(c, model, cap=1 << 4, n_random=0, n_pairs=0)
            if fails := fo.common_reference_failures(c, model, r.reference, patterns):
                bad.append(("constraints", src, fails[0], c))
        if fails := fo.simulated_failures(c, model, r.reference, rng, n_patterns=16):
            bad.append(("simulation", src, fails[0], c))
        if accepted == 30:
            break
    assert accepted == 30, f"only {accepted} accepted circuits: the property was barely exercised"
    assert not bad, "frame_css accepted a reference that some lossy circuit cannot produce: " + report(bad)


@pytest.mark.parametrize("name", WEAKER)
def test_p1_accepted_reference_holds_for_every_weaker_model(name):
    model, rng = WEAKER[name], seed("p1-weaker", name)
    bad, accepted = [], 0
    for src, c in corpus(rng, PESSIMISTIC, n_random=12, per_set=4):
        r = frame_css(c)
        if r.verdict != "frame-css":
            continue
        accepted += 1
        if fails := fo.simulated_failures(c, model, r.reference, rng, n_patterns=16):
            bad.append(("simulation", src, fails[0], c))
        if accepted == 30:
            break
    assert accepted == 30, f"only {accepted} accepted circuits: the property was barely exercised"
    assert not bad, "a pessimistic proof failed under a weaker loss model: " + report(bad)


@pytest.mark.parametrize("name", MODELS)
def test_p1_oracle_reference_is_common_too(name):
    model, rng = MODELS[name], seed("p1-oracle", name)
    bad = []
    for src, c in corpus(rng, model, n_random=8, per_set=1):
        ok, kappa = fo.brute_force_frame_css(c, model)
        if ok and fo.simulated_failures(c, model, kappa, rng, n_patterns=20):
            bad.append((src, c))
    assert not bad, "the theorem fails for the brute-force frames (oracle bug or paper bug): " + report(bad)


# ------------------------------------------------------------------ P2 completeness
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("gate_set", fo.GATE_SETS)
def test_p2_constructive_circuits_are_accepted(name, gate_set):
    model, rng = MODELS[name], seed("p2", name, gate_set)
    bad = []
    for _ in range(4):
        c = fo.random_frame_css(rng, int(rng.integers(2, 5)), int(rng.integers(4, 14)), gate_set, model, "resets")
        assert fo.brute_force_frame_css(c, model)[0], f"premise: the generator builds frame-CSS circuits\n{c}"
        r = frame_css(c)
        if r.verdict != "frame-css":
            bad.append((r.verdict, r.reasons[:1], c))
    assert not bad, "a circuit that is frame-CSS by construction was not accepted: " + report(bad)


# ------------------------------------------------------------------ P3 exactness
@pytest.mark.parametrize("name", MODELS)
def test_p3_verdict_matches_brute_force(name):
    model, rng = MODELS[name], seed("p3", name)
    bad, seen = [], {"frame-css": 0, "not-frame-css": 0}
    for src, c in corpus(rng, model, n_random=30, per_set=2):
        assert not fo.scope_violations(c), "premise: the corpus stays inside assumption (i)"
        want, got = oracle(c, model), frame_css(c).verdict
        seen[want] += 1
        if got != want:
            bad.append((src, f"frame_css={got}", f"oracle={want}", c))
    assert min(seen.values()) >= 20, f"one-sided corpus {seen}: exactness was not tested in both directions"
    assert not bad, "frame_css disagrees with the brute-force frame search: " + report(bad)


@pytest.mark.parametrize("text, verdict", [
    ("RX 0\nS 0 0\nMX 0", "frame-css"),
    ("RX 0\nS 0 0\nMX 0\nH 0", "not-frame-css"),
])
def test_p3_reload_anywhere_includes_the_end_of_the_circuit(text, verdict):
    # after MX reads -X the next frame keeps the sign; H turns it into the Z channel, read by a reload at the end
    assert oracle(stim.Circuit(text), PESSIMISTIC) == verdict
    assert frame_css(stim.Circuit(text)).verdict == verdict


# ------------------------------------------------------------------ P4 scope
OUTSIDE_I = [
    "R 0 1\nM 0\nCX rec[-1] 1\nM 1",
    "R 0 1\nM 0\nCZ 1 rec[-1]\nM 1",
    "R 0 1\nCX sweep[0] 1\nM 1",
    "R 0 1\nMPP X0*X1\nM 0 1",
    "R 0 1\nMXX 0 1\nM 0 1",
    "R 0 1\nMYY 0 1\nM 0 1",
    "R 0 1\nMZZ 0 1\nM 0 1",
    "R 0 1\nSPP X0*Z1\nM 0 1",
    "R 0 1\nSPP_DAG Y0\nM 0 1",
    "R 0\nHERALDED_ERASE(0.1) 0\nM 0",
    "R 0\nHERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0) 0\nM 0",
]
INSIDE_I = """QUBIT_COORDS(0, 0) 0
R 0 1
RX 2
RY 3
TICK
X_ERROR(0.1) 0
Y_ERROR(0.1) 0
Z_ERROR(0.1) 1
DEPOLARIZE1(0.1) 2
DEPOLARIZE2(0.1) 0 1
PAULI_CHANNEL_1(0.1, 0, 0) 0
PAULI_CHANNEL_2(0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1
E(0.1) X0 Z1
ELSE_CORRELATED_ERROR(0.1) Y2
I_ERROR(0.1) 0
II_ERROR(0.1) 0 1
I_ERROR[reload](0.5) 0
MPAD 1
CX 0 1
X 1
M(0.01) !0
MR 1
MX 2
MY 3
MRX 2
MRY !3
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]
SHIFT_COORDS(1)"""


@pytest.mark.parametrize("name", ["2q", "anywhere", "mixed"])
@pytest.mark.parametrize("text", OUTSIDE_I)
def test_p4_operations_outside_assumption_i(text, name):
    c, model = stim.Circuit(text), WEAKER[name]
    assert fo.scope_violations(c), "premise: the oracle sees an operation outside assumption (i)"
    assert frame_css(c).verdict == "outside-assumptions"
    assert certify(c, model).verdict == "outside-assumptions"


@pytest.mark.parametrize("name", MODELS)
def test_p4_every_operation_of_assumption_i_is_decided(name):
    c, model = stim.Circuit(INSIDE_I), MODELS[name]
    assert not fo.scope_violations(c), "premise: inside assumption (i)"
    assert frame_css(c).verdict == oracle(c, model)


# ------------------------------------------------------------------ P5 invariance
@pytest.mark.parametrize("name", MODELS)
def test_p5_rewrites_keep_the_verdict(name):
    model, rng = MODELS[name], seed("p5", name)
    bad, n = [], 0
    for src, c in corpus(rng, model, n_random=10, per_set=2):
        if src.count("/") > 1:
            continue
        want, got = oracle(c, model), frame_css(c).verdict
        for kind, c2 in fo.rewrites(c, rng, model).items():
            assert oracle(c2, model) == want, f"premise: {kind} keeps the definition's verdict\n{c}\n---\n{c2}"
            n += 1
            if (got2 := frame_css(c2).verdict) != got:
                bad.append((src, kind, f"{got} -> {got2}", f"{c}\n--- rewritten:\n{c2}"))
    assert n >= 100, f"only {n} rewrites tried"
    assert not bad, "a verdict-preserving rewrite changed frame_css's verdict: " + report(bad)


# ------------------------------------------------------------------ P6 contradictions
@pytest.mark.parametrize("name", ["2q", "any", "every", "reset"])
def test_p6_proven_contradiction_excludes_frame_css(name):
    model, rng = WEAKER[name], seed("p6", name)
    reload = model.reload == "reset"
    bad, contradictions = [], 0
    for src, c in corpus(rng, model, n_random=120, per_set=1):
        fam = loss_family(c, reload=reload, slots=loss_locations(c, model), n_random=10, seed=0)
        if search(c, fam, reload)[1] is None:
            continue
        contradictions += 1
        if frame_css(c).verdict == "frame-css" or certify(c, model, n_random=10).verdict == "proven":
            bad.append((src, c))
    assert contradictions >= 5, f"only {contradictions} contradictions found: the property was barely exercised"
    assert not bad, "frame_css accepted a circuit whose lossy circuits contradict each other: " + report(bad)
