"""Exact phase-guard references and the sampler's per-shot loss integration, each checked against the exact sign
constraints of the lossy circuit, with and without reload markers at the resets (`reload`)."""
import json
from pathlib import Path

import frame_oracle as fo
import numpy as np
import pytest
import stim
from ellis.certify import constraints
from ellis.gf2 import pack
from ellis.msc.reference import CultivationReference

from ellis import _core
from ellis import frame as mf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def valid(circuit, losses, reference):
    record = pack(reference)
    return all((y & record).bit_count() % 2 == v for y, v in constraints(circuit, losses))


def with_reloads(circuit, reload):
    """(circuit, move): the circuit itself, or with reload markers at its resets; move re-keys loss patterns."""
    return fo.reload_at_resets(circuit) if reload else (circuit, lambda s: s)


@pytest.mark.parametrize("reload", [False, True])
def test_native_fallback_preserves_canonical_reference(reload):
    circuit, move = with_reloads(stim.Circuit("RX 0 1\nS 0 0\nCZ 0 1\nMY !0\nMRY 0\nMY 0\nRX 1\nMRX !1\nMPAD 1 0"), reload)
    skeleton = mf.reference_skeleton(circuit)
    plan = _core.Plan(skeleton, circuit.num_qubits)
    patterns = [move(s) for s in [{}, {-1: [0]}, {2: [0]}, {2: [0, 1]}, {2: [0], 5: [0]}, {3: [1]}]]
    batch = plan.batch_reference(patterns)
    for pattern, record in zip(patterns, batch):
        record = np.frombuffer(record, dtype=np.bool_)
        assert np.array_equal(record, mf.lossy_reference(skeleton, pattern))
        assert valid(circuit, pattern, record)


def toy(phases="S 0"):
    circuit = stim.Circuit(f"RX 0 1\n{phases}\nCX 0 1")
    # Terminal nondemolition YY, XX, ZZ readouts in the cultivation convention.
    boundary = stim.Circuit("H_YZ 0 1\nCX 0 1\nM 1\nCX 0 1\nH_YZ 1 0\n"
                            "H 0 1\nCX 0 1\nM 1\nCX 0 1\nH 1 0\nCX 0 1\nM 1\nCX 0 1")
    for inst in boundary:
        assert isinstance(inst, stim.CircuitInstruction)
        circuit.append(inst.name, inst.targets_copy(), inst.gate_args_copy(), tag="noiseless")
    return circuit


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("phases", ["S 0", "S_DAG 0", "S 0 0"])
def test_fast_reference_gauge_and_terminal_phase(reload, phases):
    circuit, move = with_reloads(toy(phases), reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    for pattern in map(move, [{}, {2: [0]}, {2: [1]}, {-1: [0]}]):
        before = provider.stats["fast"]
        reference = provider.get(pattern)
        assert provider.stats["fast"] == before + 1
        assert valid(circuit, pattern, reference)
    if phases == "S 0 0":
        assert provider.get({}).tolist() == [False, True, False]


@pytest.mark.parametrize("where", ["suffix", "after-boundary"])
def test_reload_marker_in_the_css_suffix_is_fast_and_after_the_boundary_falls_back(where):
    # toy("S 0"): 0 RX, 1 S, 2 CX (the cut), 3.. the noiseless queries; 5 is the first query's readout of qubit 1
    base = toy("S 0").flattened()
    k = 3 if where == "suffix" else 6
    circuit = base[:k]
    circuit.append("I_ERROR", [0, 1] if where == "suffix" else [1], tag="reload")
    circuit += base[k:]
    provider = CultivationReference(circuit)
    meta = provider.metadata
    assert meta is not None and (meta["cut"] <= k < meta["boundary"]) == (where == "suffix") and k > meta["cut"]
    patterns = [{}, {2: [0]}, {2: [1]}, {2: [0, 1]}, {-1: [0]}, {-1: [1]}]
    for pattern, reference in zip(patterns, provider.get_many(patterns)):
        assert valid(circuit, pattern, reference), pattern
    path = "fast" if where == "suffix" else "fallback"
    assert provider.stats == {"queries": len(patterns), "fast": 0, "fallback": 0, path: len(patterns)}


def test_two_independent_y_directions_require_fallback():
    circuit = toy("S 0 1")
    provider = CultivationReference(circuit)
    reference = provider.get({})
    assert provider.stats == {"queries": 1, "fast": 0, "fallback": 1}
    assert valid(circuit, {}, reference)


@pytest.mark.parametrize("reload", [False, True])
def test_random_loss_sampling_uses_the_actual_pattern(reload, monkeypatch):
    circuit = stim.Circuit()
    for inst in toy("S_DAG 0"):
        circuit.append(inst)
        if inst.name in mf.UNITARY_2Q and inst.tag != "noiseless":
            circuit.append("I_ERROR", inst.targets_copy(), [0.4], tag="loss")
    circuit, _ = with_reloads(circuit, reload)
    provider = CultivationReference(circuit)
    captured, original = [], provider.get_many

    def capture(patterns):
        captured.extend(patterns)
        return original(patterns)

    monkeypatch.setattr(provider, "get_many", capture)
    measurements, heralds = mf.sample(circuit, 128, seed=17, reference_provider=provider)
    assert len(captured) == 128 and heralds.any()
    equations = {}
    for pattern, record in zip(captured, measurements):
        key = json.dumps(pattern, sort_keys=True)
        if key not in equations:
            equations[key] = constraints(circuit, pattern)
        packed = pack(record)
        assert all((y & packed).bit_count() % 2 == v for y, v in equations[key])
    assert provider.stats == {"queries": 128, "fast": 128, "fallback": 0}


def append_terminal_query(circuit, qubits, bases):
    """Independent fan-in expansion of one positive nondemolition Pauli query."""
    pivot = qubits[-1]
    compute = []
    for q, basis in zip(qubits, bases):
        if basis != "Z":
            compute.append(("H" if basis == "X" else "H_YZ", [q]))
    compute += [("CX", [q, pivot]) for q in qubits[:-1]]
    for name, targets in compute:
        circuit.append(name, targets, tag="noiseless")
    circuit.append("M", [pivot], tag="noiseless")
    for name, targets in reversed(compute):
        circuit.append(name, targets, tag="noiseless")


@pytest.mark.parametrize("reload", [False, True])
def test_terminal_effect_guard_accepts_noncommuting_individual_queries(reload):
    circuit = stim.Circuit("RX 0\nS 0")
    for bases in ("Y", "X", "Z"):
        append_terminal_query(circuit, [0], bases)
    # Q=Y anticommutes with both later queries. Their backward effect is
    # Pi_X Pi_Z Pi_X = Pi_X/2, whose Q-commuting subgroup is trivial.
    assert not stim.PauliString("Y").commutes(stim.PauliString("X"))
    assert not stim.PauliString("Y").commutes(stim.PauliString("Z"))
    circuit, _ = with_reloads(circuit, reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    reference = provider.get({})
    assert provider.stats == {"queries": 1, "fast": 1, "fallback": 0}
    assert reference.tolist() == [False, False, False]
    assert valid(circuit, {}, reference)


@pytest.mark.parametrize("reload", [False, True])
def test_terminal_effect_guard_rejects_both_anticommuting_families(reload):
    circuit = stim.Circuit("RX 0 1\nS 1")
    append_terminal_query(circuit, [0], "Y")
    append_terminal_query(circuit, [0, 1], "XX")
    append_terminal_query(circuit, [0, 1], "ZZ")
    # XX and ZZ commute, so both survive in the backward effect. Q=Y0
    # anticommutes with each, and choosing all three outputs zero is impossible
    # for input X0,Y1: after Q=+Y0, YY=+1 requires XX xor ZZ = 1.
    assert stim.PauliString("XX").commutes(stim.PauliString("ZZ"))
    circuit, _ = with_reloads(circuit, reload)
    assert not valid(circuit, {}, np.zeros(circuit.num_measurements, dtype=bool))
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    reference = provider.get({})
    assert provider.stats == {"queries": 1, "fast": 0, "fallback": 1}
    assert valid(circuit, {}, reference)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("reload", [False, True])
def test_end_to_end_cultivation_witnesses(distance, reload):
    stem = f"msc_e2e_d{distance}_d15_r3_r5"
    mode = "reload" if reload else "no_reload"
    circuit, move = with_reloads(stim.Circuit.from_file(FIXTURES / "cultivation" / f"{stem}.stim"), reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    assert len(provider.metadata["core_qubits"]) == (15 if distance == 3 else 42)
    # the witnesses of the removed reload=True mode, re-keyed to the circuit with reload markers
    witness = json.loads((FIXTURES / "cultivation" / f"{stem}__{mode}__transferred.json").read_text())
    patterns = [move({int(k): qs for k, qs in pattern.items()}) for pattern, _, _ in witness["witness"]]
    references = provider.get_many(patterns)
    for pattern, reference in zip(patterns, references):
        assert valid(circuit, pattern, reference)
    pattern = patterns[-1]
    measurements, _ = mf.sample(circuit, 64, forced_losses=pattern, reference_provider=provider)
    rows = constraints(circuit, pattern)
    assert all(all((y & pack(record)).bit_count() % 2 == v for y, v in rows) for record in measurements)


@pytest.mark.parametrize("reload", [False, True])
def test_general_clifford_prefix_and_css_tail(reload):
    circuit, move = with_reloads(stim.Circuit("RY 0\nH 1\nCZ[noiseless] 1 2\nCX 0 1\nM !1\n"
                                              "RX 2\nCX 2 3\nMRX 0\nM 2 3\nMPAD 1"), reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    assert provider.metadata["core_qubits"] == [0, 1, 2]
    assert provider.metadata["queries"] == []
    patterns = [move(s) for s in [{}, {-1: [0]}, {1: [1]}, {2: [2]}, {3: [0, 1]}, {5: [2]}, {6: [3]}]]
    for pattern in patterns:
        assert valid(circuit, pattern, provider.get(pattern))
    assert provider.stats["fast"] == len(patterns)


@pytest.mark.parametrize("reload", [False, True])
def test_general_prefix_guard_can_reject(reload):
    circuit, move = with_reloads(stim.Circuit("RY 0\nRX 1\nCZ 0 1\nCX 0 1\nMX 0 1"), reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    assert valid(circuit, {}, provider.get({}))
    assert provider.stats["fallback"] == 1
    pattern = move({1: [1]})
    assert valid(circuit, pattern, provider.get(pattern))
    assert provider.stats["fast"] == 1


def test_short_css_readout_tail_uses_native_tableau():
    circuit = stim.Circuit("RX 0 1\nS 0\nCZ 0 1\nMX 0 1")
    provider = CultivationReference(circuit)
    assert provider.metadata is None
    assert valid(circuit, {}, provider.get({}))


@pytest.mark.parametrize("reload", [False, True])
def test_css_only_needs_no_prefix_qubits(reload):
    circuit, move = with_reloads(stim.Circuit("RX 0\nCX 0 1\nM !1\nMRX 0\nR 1\nM 0 1\nMPAD 1 0"), reload)
    provider = CultivationReference(circuit)
    assert provider.metadata is not None
    assert provider.metadata["core_qubits"] == []
    for pattern in map(move, [{}, {-1: [0]}, {1: [0, 1]}, {2: [1]}]):
        assert valid(circuit, pattern, provider.get(pattern))
    assert provider.stats["fallback"] == 0


@pytest.mark.parametrize("reload", [False, True])
def test_general_prefix_gate_vocabulary(reload):
    rng = np.random.default_rng(91)
    gates = ["H", "S", "S_DAG", "SQRT_X", "C_XYZ", "RY", "MY", "MRY", "CZ", "CY", "ISWAP", "SWAP"]
    counts = {"fast": 0, "fallback": 0}
    for _ in range(24):
        circuit = stim.Circuit()
        for _ in range(8):
            name = str(rng.choice(gates))
            width = 2 if name in mf.UNITARY_2Q else 1
            circuit.append(name, rng.choice(4, size=width, replace=False).tolist())
        circuit.append("S", [0])
        end = len(circuit) - 1
        circuit += stim.Circuit("CX 0 1 2 3\nMRX !0 2\nR 1\nCX 1 3\nM 0 1 2 3")
        circuit, move = with_reloads(circuit, reload)
        provider = CultivationReference(circuit)
        assert provider.metadata is not None
        for pattern in map(move, [{}, {-1: [int(rng.integers(4))]},
                                  {end: [int(rng.integers(4))], end + 2: [int(rng.integers(4))]}]):
            assert valid(circuit, pattern, provider.get(pattern))
        for key in counts:
            counts[key] += provider.stats[key]
    assert counts["fast"] and counts["fallback"]


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("manifest", ["benchmark_2q.jsonl", "sat_2q.jsonl", "sat_ls_2q.jsonl"])
def test_general_provider_on_saved_contextual_witnesses(reload, manifest):
    for line in (FIXTURES / "certify" / manifest).read_text().splitlines():
        row = json.loads(line)
        if row["verdict"] != "contextual":
            continue
        circuit, move = with_reloads(stim.Circuit.from_file(FIXTURES / "lightstim" / (row["circuit"] + ".stim")), reload)
        provider = CultivationReference(circuit)
        for saved in row["witness_patterns"]:
            pattern = move({int(k): qs for k, qs in saved.items()})
            assert valid(circuit, pattern, provider.get(pattern)), row["circuit"]
