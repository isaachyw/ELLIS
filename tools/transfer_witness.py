"""Transfer a pre-boundary cultivation witness to its end-to-end circuit.

Physical two-qubit gates must be a coordinate-matched prefix on each wire;
disjoint gates may be reordered. Measurement records are matched by coordinate,
gate, and occurrence on the same qubit.
The transferred witness is checked with exact flows and independent sampling.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import stim
from ellis.certify import constraints, loss_slots, verify_witness

from ellis import frame as mf


def physical_pairs(circuit):
    coords = circuit.get_final_qubit_coordinates()
    pairs, sites = [], {}
    for k, inst in enumerate(circuit):
        if inst.name not in mf.UNITARY_2Q or inst.tag == "noiseless":
            continue
        qs = [t.value for t in inst.targets_copy()]
        assert len(qs) == len(set(qs)), "repeated operands within a 2q instruction"
        for a, b in zip(qs[::2], qs[1::2]):
            ordinal = len(pairs)
            pairs.append({"key": (inst.name, tuple(coords[a]), tuple(coords[b])),
                          "instruction": k, "qubits": (a, b)})
            sites[k, a], sites[k, b] = (ordinal, 0), (ordinal, 1)
    return pairs, sites


def measurement_keys(circuit):
    coords = circuit.get_final_qubit_coordinates()
    occurrences, keys = Counter(), []
    for inst in circuit:
        if inst.name == "MPAD":
            raise ValueError("MPAD measurement matching is not implemented")
        if inst.name not in mf.MEAS:
            continue
        for target in inst.targets_copy():
            coord = tuple(coords[target.value])
            keys.append((coord, inst.name, occurrences[coord], target.is_inverted_result_target))
            occurrences[coord] += 1
    assert len(keys) == circuit.num_measurements
    return keys


def parity_value(rows, parity):
    pivots = {}
    for y, value in rows:
        while y:
            high = y.bit_length() - 1
            if high not in pivots:
                pivots[high] = (y, value)
                break
            py, pv = pivots[high]
            y ^= py; value ^= pv
    y, value = sum(1 << j for j in parity), 0
    while y:
        high = y.bit_length() - 1
        assert high in pivots, "mapped parity is not deterministic in the target circuit"
        py, pv = pivots[high]
        y ^= py; value ^= pv
    return value


def transfer(source, target, witness):
    source_pairs, source_sites = physical_pairs(source)
    target_pairs, _ = physical_pairs(target)
    source_wires, target_wires = {}, {}
    for pairs, wires in ((source_pairs, source_wires), (target_pairs, target_wires)):
        for pair in pairs:
            for coord in pair["key"][1:]:
                wires.setdefault(coord, []).append(pair["key"])
    for coord, sequence in source_wires.items():
        assert sequence == target_wires.get(coord, [])[:len(sequence)], (
            f"physical 2q order differs on wire {coord}")
    target_occurrences, seen = {}, Counter()
    for ordinal, pair in enumerate(target_pairs):
        key = pair["key"]
        target_occurrences[key, seen[key]] = ordinal
        seen[key] += 1
    pair_map, seen = {}, Counter()
    for ordinal, pair in enumerate(source_pairs):
        key = pair["key"]
        pair_map[ordinal] = target_occurrences[key, seen[key]]
        seen[key] += 1
    source_records = measurement_keys(source)
    target_records = {key: j for j, key in enumerate(measurement_keys(target))}
    allowed = set(loss_slots(target, "2q", False))
    mapped, cache = [], {}
    for pattern, parity, expected in witness:
        losses = {}
        for k, qubits in pattern.items():
            for q in qubits:
                ordinal, operand = source_sites[int(k), q]
                pair = target_pairs[pair_map[ordinal]]
                slot = pair["instruction"], pair["qubits"][operand]
                assert slot in allowed
                losses.setdefault(slot[0], []).append(slot[1])
        losses = {k: sorted(set(qs)) for k, qs in sorted(losses.items())}
        indices = [target_records[source_records[j]] for j in parity]
        key = json.dumps(losses, sort_keys=True)
        if key not in cache:
            cache[key] = constraints(target, losses)
        assert parity_value(cache[key], indices) == expected, "mapped parity has the wrong value"
        mapped.append((losses, indices, expected))
    verify_witness(target, mapped, shots=128)
    return mapped, len(source_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_certificate", type=Path)
    parser.add_argument("target_circuit", type=Path)
    parser.add_argument("--source-circuit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cultivation")
    args = parser.parse_args()
    certificate = json.loads(args.source_certificate.read_text())
    assert certificate["verdict"] == "contextual", "source certificate is not contextual"
    assert not certificate.get("reload"), ("the implicit reload at resets is gone: certify a circuit with "
                                           "I_ERROR[reload] markers instead")
    source_path = args.source_circuit
    source_text, target_text = source_path.read_text(), args.target_circuit.read_text()
    source, target = stim.Circuit(source_text).flattened(), stim.Circuit(target_text).flattened()
    mapped, count = transfer(source, target, certificate["witness"])
    row = {"circuit": args.target_circuit.stem, "reload": False,
           "verdict": "contextual", "method": "transferred-stage12-witness",
           "placement": "2q", "include_noiseless": False, "max_losses": None,
           "witness": mapped, "witness_verified_shots": 128, "witness_flow_verified": True,
           "source_certificate": str(args.source_certificate), "source_circuit": str(source_path),
           "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
           "target_sha256": hashlib.sha256(target_text.encode()).hexdigest(),
           "matched_physical_2q_pairs": count,
           "mapping": "directed-pair occurrences; physical 2q order is preserved on each source wire"}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"{args.target_circuit.stem}__no_reload__transferred.json"
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({"output": str(output), "verdict": row["verdict"],
                      "witness_rows": len(mapped), "reload": row["reload"]}), flush=True)


if __name__ == "__main__":
    main()
