"""Export real_sim's noiseless Clifford cultivation benchmarks for loss studies.

Boundary MPPs use real_sim's rotation/CX/M/uncompute expansion. Every generated
gate is tagged ``noiseless`` so virtual boundary gates are not loss locations.
Gate masking of this expansion follows real_sim's convention; it does not define
an arbitrary native MPP measurement on absent atoms. No coherent R_Z is added.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import stim


def expand_mpp(circuit):
    out = stim.Circuit()
    rotations = {"X": "H", "Y": "H_YZ", "Z": None}
    for inst in circuit.flattened():
        if inst.name != "MPP":
            out.append(inst)
            continue
        for group in inst.target_groups():
            terms = []
            for target in group:
                if target.is_inverted_result_target:
                    raise ValueError("inverted MPP targets are not supported")
                basis = "X" if target.is_x_target else "Y" if target.is_y_target else "Z"
                terms.append((basis, target.qubit_value))
            changes = []
            for basis, qubit in terms:
                gate = rotations[basis]
                if gate is not None:
                    changes.append((gate, qubit))
            accumulator = terms[-1][1]
            for gate, qubit in changes:
                out.append(gate, [qubit], tag="noiseless")
            for _, qubit in terms[:-1]:
                out.append("CX", [qubit, accumulator], tag="noiseless")
            out.append("M", [accumulator], tag="noiseless")
            for _, qubit in reversed(terms[:-1]):
                out.append("CX", [qubit, accumulator], tag="noiseless")
            for gate, qubit in reversed(changes):
                out.append(gate, [qubit], tag="noiseless")
    return out


def metadata(original, exported, options):
    pairs = Counter()
    for inst in original.flattened():
        if stim.gate_data(inst.name).is_two_qubit_gate:
            targets = inst.targets_copy()
            pairs[inst.name] += sum(
                targets[k].is_qubit_target and targets[k + 1].is_qubit_target
                for k in range(0, len(targets), 2)
            )
    return {
        "options": options,
        "num_qubits": exported.num_qubits,
        "num_measurements": exported.num_measurements,
        "num_detectors": exported.num_detectors,
        "num_observables": exported.num_observables,
        "num_instructions_flat": len(exported.flattened()),
        "original_gate_instruction_counts": dict(Counter(i.name for i in original.flattened())),
        "exported_gate_instruction_counts": dict(Counter(i.name for i in exported.flattened())),
        "physical_2q_pair_counts": dict(pairs),
        "physical_2q_loss_sites": 2 * sum(pairs.values()),
        "original_sha256": hashlib.sha256(str(original).encode()).hexdigest(),
        "exported_sha256": hashlib.sha256(str(exported).encode()).hexdigest(),
        "boundary_convention": "MPP expanded as in real_sim; all generated gates tagged noiseless",
        "model": "Clifford Y-basis surrogate; no coherent R_Z or Pauli noise",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-sim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cultivation")
    args = parser.parse_args()
    vendor_src = args.real_sim_root / "vendor/magic_state_cultivation/src"
    sys.path.insert(0, str(vendor_src))
    integration = import_module("cultiv._construction._integration")

    args.output.mkdir(parents=True, exist_ok=True)
    for dcolor in (3, 5):
        common = {"dcolor": dcolor, "inject_style": "unitary", "basis": "Y"}
        for stage in ("stage12", "e2e"):
            options = dict(common)
            if stage == "stage12":
                original = integration.make_inject_and_cultivate_circuit(**options)
                name = f"msc_stage12_d{dcolor}"
            else:
                options.update(dsurface=15, r_growing=3, r_end=5)
                original = integration.make_end2end_cultivation_circuit(**options)
                name = f"msc_e2e_d{dcolor}_d15_r3_r5"
            exported = expand_mpp(original)
            row = metadata(original, exported, {"stage": stage, **options})
            (args.output / f"{name}.stim").write_text(str(exported))
            (args.output / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(f"{name}: {row['num_qubits']} qubits, "
                  f"{row['physical_2q_loss_sites']} physical 2q loss sites", flush=True)


if __name__ == "__main__":
    main()
