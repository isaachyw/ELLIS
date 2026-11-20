"""frame_css against independent oracles (frame_oracle.py), across loss models: an accepted reference is common to
every loss pattern (P1), circuits that are frame-CSS by construction are accepted (P2), the verdict equals a
brute-force frame search (P3), operations outside assumption (i) and prefixes beyond the choice cap are
"outside-assumptions" (P4), verdict-preserving rewrites keep the verdict (P5), and a proven contradiction
excludes frame-CSS (P6)."""
import sys

import numpy as np
import pytest
import stim
from ellis.certify import LossModel, certify, frame_css, loss_family, loss_locations, search

import frame_oracle as fo

MODELS = {
    "2q": LossModel(),
    "2q-start": LossModel(from_start=True),
    "any": LossModel(after="any"),
    "any-tags-lossy": LossModel(after="any", noiseless_loss_free=False),
    "every": LossModel(after="any", noiseless_loss_free=False, from_start=True),
    "anywhere": LossModel(after="any", reload="anywhere"),
    "anywhere-2q": LossModel(reload="anywhere"),
    "mixed": LossModel(after="any", reload_state="mixed"),
    "mixed-anywhere": LossModel(after="any", reload="anywhere", reload_state="mixed"),
}
ZERO = [name for name, m in MODELS.items() if m.reload_state == "zero"]


def seed(*keys) -> np.random.Generator:
    return np.random.default_rng([sum(map(ord, str(k))) for k in keys])


def oracle(c: stim.Circuit, model: LossModel) -> str:
    return "frame-css" if fo.brute_force_frame_css(c, model)[0] else "not-frame-css"


def encoder_allowed(model: LossModel) -> bool:
    return model.noiseless_loss_free and not model.from_start


def constructive(rng, model: LossModel, per_set: int):
    for gate_set in fo.GATE_SETS:
        for prefix in ("resets", "encoder") if encoder_allowed(model) else ("resets",):
            for _ in range(per_set):
                n = int(rng.integers(2, 5))
                yield f"{gate_set}/{prefix}", fo.random_frame_css(rng, n, int(rng.integers(4, 12)), gate_set, model, prefix)


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
    for src, c in corpus(rng, model, n_random=12, per_set=2):
        r = frame_css(c, model)
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
    for prefix in ("resets", "encoder") if encoder_allowed(model) else ("resets",):
        for _ in range(4):
            c = fo.random_frame_css(rng, int(rng.integers(2, 5)), int(rng.integers(4, 14)), gate_set, model, prefix)
            assert fo.brute_force_frame_css(c, model)[0], f"premise: the generator builds frame-CSS circuits\n{c}"
            r = frame_css(c, model)
            if r.verdict != "frame-css":
                bad.append((prefix, r.verdict, r.reasons[:1], c))
    assert not bad, "a circuit that is frame-CSS by construction was not accepted: " + report(bad)


# ------------------------------------------------------------------ P3 exactness
@pytest.mark.parametrize("name", MODELS)
def test_p3_verdict_matches_brute_force(name):
    model, rng = MODELS[name], seed("p3", name)
    bad, seen = [], {"frame-css": 0, "not-frame-css": 0}
    for src, c in corpus(rng, model, n_random=30, per_set=2):
        assert not fo.scope_violations(c), "premise: the corpus stays inside assumption (i)"
        want, got = oracle(c, model), frame_css(c, model).verdict
        seen[want] += 1
        if got != want:
            bad.append((src, f"frame_css={got}", f"oracle={want}", c))
    assert min(seen.values()) >= 20, f"one-sided corpus {seen}: exactness was not tested in both directions"
    assert not bad, "frame_css disagrees with the brute-force frame search: " + report(bad)


@pytest.mark.parametrize("text, name", [
    ("R 0 1\nCX 0 1\nI_ERROR[reload] 0\nSQRT_X 0 0\nI_ERROR[reload] 0\nM 0 1", "2q"),
    ("R 0 1\nCX 0 1\nI_ERROR[reload] 0\nSQRT_X 0 0\nM 0 1", "anywhere-2q"),
])
def test_p3_a_certain_reload_refills_the_slot_for_good(text, name):
    # qubit 0's only loss location precedes a reload marker with p = 1, so it is present at every later point:
    # no reload can happen after the marker, and SQRT_X SQRT_X may flip its frame there
    c, model = stim.Circuit(text), MODELS[name]
    ok, kappa = fo.brute_force_frame_css(c, model)
    assert ok and kappa.tolist() == [True, False], "premise: frame-CSS by the definition"
    assert fo.simulated_failures(c, model, kappa, seed("refill"), n_patterns=200) == [], "premise: kappa is common"
    assert frame_css(c, model).verdict == "frame-css"


def test_p3_a_prefix_reset_leaves_its_partner_maximally_mixed():
    # the noiseless prefix resets one half of a Bell pair: the other half is maximally mixed, so the reload
    # marker's rule 4 is the only condition on its frame; the verdict must not depend on a random branch
    c = stim.Circuit("RX[noiseless] 0\nR[noiseless] 1\nCX[noiseless] 0 1\nR[noiseless] 0\nI 1\nI_ERROR[reload] 1\nM 1")
    model = MODELS["any"]
    assert oracle(c, model) == "frame-css", "premise: frame-CSS by the definition"
    assert {frame_css(c, model).verdict for _ in range(40)} == {"frame-css"}


@pytest.mark.parametrize("text, verdict", [
    ("RX 0\nS 0 0\nMX 0", "frame-css"),
    ("RX 0\nS 0 0\nMX 0\nH 0", "not-frame-css"),
])
def test_p3_reload_anywhere_includes_the_end_of_the_circuit(text, verdict):
    # after MX reads -X the next frame keeps the sign; H turns it into the Z channel, read by a reload at the end
    model = MODELS["anywhere"]
    assert oracle(stim.Circuit(text), model) == verdict
    assert frame_css(stim.Circuit(text), model).verdict == verdict


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


@pytest.mark.parametrize("name", ["2q", "every", "anywhere", "mixed"])
@pytest.mark.parametrize("text", OUTSIDE_I)
def test_p4_operations_outside_assumption_i(text, name):
    c, model = stim.Circuit(text), MODELS[name]
    assert fo.scope_violations(c), "premise: the oracle sees an operation outside assumption (i)"
    assert frame_css(c, model).verdict == "outside-assumptions"
    assert certify(c, model).verdict == "outside-assumptions"


@pytest.mark.parametrize("name", MODELS)
def test_p4_every_operation_of_assumption_i_is_decided(name):
    c, model = stim.Circuit(INSIDE_I), MODELS[name]
    assert not fo.scope_violations(c), "premise: inside assumption (i)"
    assert frame_css(c, model).verdict == oracle(c, model)


RING = "H[noiseless] 0 1 2 3 4\nCZ[noiseless] 0 1 1 2 2 3 3 4 4 0\n"
GHZ = "H[noiseless] 5\nCX[noiseless] 5 6 6 7 7 8 8 9 9 10 10 11 11 12\n"


def read_all(n: int) -> str:
    qs = " ".join(map(str, range(n)))
    return f"I {qs}\nM {qs}"


def test_p4_prefix_with_more_choices_than_the_cap(monkeypatch):
    # a 5-ring graph state (CSS in no frames) next to an 8-qubit GHZ state: more prefix choices than the cap
    monkeypatch.setattr(sys.modules["ellis.certify.frame_css"], "PREFIX_CHOICES", 256)
    c, model = stim.Circuit(RING + GHZ + read_all(13)), MODELS["any"]
    r = frame_css(c, model)
    assert r.verdict == "outside-assumptions" and r.reasons[0].startswith("assumption (ii)"), r.reasons
    assert certify(c, model).verdict == "outside-assumptions"


def test_p4_prefix_within_the_cap_is_decided_exactly():
    c, model = stim.Circuit(RING + read_all(5)), MODELS["any"]
    assert oracle(c, model) == "not-frame-css", "premise: the 5-ring graph state is CSS in no frames"
    assert frame_css(c, model).verdict == "not-frame-css"


# ------------------------------------------------------------------ P5 invariance
@pytest.mark.parametrize("name", MODELS)
def test_p5_rewrites_keep_the_verdict(name):
    model, rng = MODELS[name], seed("p5", name)
    bad, n = [], 0
    for src, c in corpus(rng, model, n_random=10, per_set=2):
        if src.count("/") > 1:
            continue
        want, got = oracle(c, model), frame_css(c, model).verdict
        for kind, c2 in fo.rewrites(c, rng, model).items():
            assert oracle(c2, model) == want, f"premise: {kind} keeps the definition's verdict\n{c}\n---\n{c2}"
            n += 1
            if (got2 := frame_css(c2, model).verdict) != got:
                bad.append((src, kind, f"{got} -> {got2}", f"{c}\n--- rewritten:\n{c2}"))
    assert n >= 100, f"only {n} rewrites tried"
    assert not bad, "a verdict-preserving rewrite changed frame_css's verdict: " + report(bad)


# ------------------------------------------------------------------ P6 contradictions
@pytest.mark.parametrize("name", ["2q", "any", "every"])
def test_p6_proven_contradiction_excludes_frame_css(name):
    model, rng = MODELS[name], seed("p6", name)
    bad, contradictions = [], 0
    for src, c in corpus(rng, model, n_random=40, per_set=1):
        fam = loss_family(c, slots=loss_locations(c, model), n_random=10, seed=0)
        if search(c, fam)[1] is None:
            continue
        contradictions += 1
        if frame_css(c, model).verdict == "frame-css" or certify(c, model, n_random=10).verdict == "proven":
            bad.append((src, c))
    assert contradictions >= 5, f"only {contradictions} contradictions found: the property was barely exercised"
    assert not bad, "frame_css accepted a circuit whose lossy circuits contradict each other: " + report(bad)


# ------------------------------------------------------------------ reload markers at every reset
def test_p1_p3_p6_with_reload_markers_at_every_reset():
    # the removed reload="reset" model, written as markers (frame_oracle.reload_at_resets)
    model, rng = MODELS["any"], seed("reset-markers")
    bad, seen, contradictions = [], {"frame-css": 0, "not-frame-css": 0}, 0
    for src, c in corpus(rng, model, n_random=30, per_set=1):
        c, _ = fo.reload_at_resets(c)
        r, want = frame_css(c, model), oracle(c, model)
        seen[want] += 1
        if r.verdict != want:
            bad.append(("P3", src, f"frame_css={r.verdict}", f"oracle={want}", c))
        if r.verdict == "frame-css" and fo.common_reference_failures(
                c, model, r.reference, fo.loss_patterns(c, model, cap=1 << 6, n_random=0, n_pairs=20)):
            bad.append(("P1", src, c))
        if search(c, loss_family(c, slots=loss_locations(c, model), n_random=10, seed=0))[1] is not None:
            contradictions += 1
            if r.verdict == "frame-css":
                bad.append(("P6", src, c))
    assert min(seen.values()) >= 10 and contradictions >= 5, (seen, contradictions)
    assert not bad, "frame_css is wrong on a circuit with reload markers at its resets: " + report(bad)


@pytest.mark.parametrize("gate_set", fo.GATE_SETS)
def test_p2_with_reload_markers_at_every_reset(gate_set):
    # A marker right before a reset loads an atom the reset discards unread, so it adds no reload rule: writing the
    # removed reload-at-reset as markers keeps every constructive circuit frame-CSS.
    model, rng = MODELS["any"], seed("p2-reset-markers", gate_set)
    bad = []
    for prefix in ("resets", "encoder"):
        for _ in range(12):
            c = fo.random_frame_css(rng, int(rng.integers(2, 5)), int(rng.integers(4, 14)), gate_set, model, prefix)
            marked, _ = fo.reload_at_resets(c)
            assert fo.brute_force_frame_css(marked, model)[0], f"premise: still frame-CSS with markers\n{marked}"
            r = frame_css(marked, model)
            if r.verdict != "frame-css":
                bad.append((prefix, r.verdict, r.reasons[:1], marked))
            elif fo.common_reference_failures(marked, model, r.reference,
                                              fo.loss_patterns(marked, model, cap=1 << 6, n_random=0, n_pairs=20)):
                bad.append(("P1", prefix, marked))
    assert not bad, "markers at the resets cost frame-CSS: " + report(bad)


def test_a_marker_is_a_reload_point_only_if_its_atom_is_read():
    # qubit 0's frame after SQRT_X_DAG cannot hold |0>, but the reset right after the marker discards the atom
    model = MODELS["any"]
    unread = stim.Circuit("RX 0\nSWAP 2 0\nSQRT_X_DAG 0\nI_ERROR[reload] 0\nRX 0\nCX 0 1\nMX 0\nM 1 2")
    assert frame_css(unread, model).verdict == oracle(unread, model) == "frame-css"
    read = stim.Circuit("RX 0\nSWAP 2 0\nSQRT_X_DAG 0\nI_ERROR[reload] 0\nCX 0 1\nRX 0\nMX 0\nM 1 2")
    r = frame_css(read, model)                           # the CX reads the reloaded atom before the reset
    assert r.verdict == oracle(read, model) == "not-frame-css" and "reload of qubit 0" in " ".join(r.reasons)
