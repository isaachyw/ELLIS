#include "ellis/frame/program.h"

#include <stdexcept>
#include <string>

#include "stim/gates/gates.h"
#include "stim/simulators/tableau_simulator.h"

namespace maskframe {
namespace {

using stim::GateType;

bool is_loss(const stim::CircuitInstruction &inst) {
    return inst.gate_type == GateType::I_ERROR && inst.tag == "loss";
}

bool is_reload(const stim::CircuitInstruction &inst) {
    return inst.gate_type == GateType::I_ERROR && inst.tag == "reload";
}

bool is_pauli_gate(GateType g) {
    return g == GateType::X || g == GateType::Y || g == GateType::Z;
}

[[noreturn]] void reject(size_t k, const stim::CircuitInstruction &inst, const std::string &why) {
    throw std::invalid_argument("instruction " + std::to_string(k) + " `" + inst.str() + "`: " + why);
}

/// How the masked frame runs an instruction; UNSUPPORTED is returned as nullopt-like `false`.
bool classify(const stim::CircuitInstruction &inst, Kind &kind) {
    switch (inst.gate_type) {
        case GateType::TICK:
        case GateType::QUBIT_COORDS:
        case GateType::SHIFT_COORDS:
        case GateType::I:
        case GateType::II:
        case GateType::II_ERROR:
            kind = Kind::SKIP;
            return true;
        case GateType::I_ERROR:
            kind = is_loss(inst) ? Kind::LOSS : is_reload(inst) ? Kind::RELOAD : Kind::SKIP;
            return true;
        case GateType::DETECTOR:
        case GateType::OBSERVABLE_INCLUDE:
        // Single-qubit Cliffords: the frame update ignores signs, exactly as in stim::FrameSimulator.
        case GateType::H:
        case GateType::H_XY:
        case GateType::H_YZ:
        case GateType::H_NXY:
        case GateType::H_NXZ:
        case GateType::H_NYZ:
        case GateType::S:
        case GateType::S_DAG:
        case GateType::SQRT_X:
        case GateType::SQRT_X_DAG:
        case GateType::SQRT_Y:
        case GateType::SQRT_Y_DAG:
        case GateType::C_XYZ:
        case GateType::C_NXYZ:
        case GateType::C_XNYZ:
        case GateType::C_XYNZ:
        case GateType::C_ZYX:
        case GateType::C_NZYX:
        case GateType::C_ZNYX:
        case GateType::C_ZYNX:
        // Single-qubit noise: applying it to a lost qubit's (never read) frame changes nothing.
        case GateType::X_ERROR:
        case GateType::Y_ERROR:
        case GateType::Z_ERROR:
        case GateType::DEPOLARIZE1:
        case GateType::PAULI_CHANNEL_1:
            kind = Kind::STIM;
            return true;
        case GateType::X:
        case GateType::Y:
        case GateType::Z:
            kind = Kind::PAULI;
            return true;
        case GateType::CX:
        case GateType::CY:
        case GateType::CZ:
        case GateType::XCX:
        case GateType::XCY:
        case GateType::XCZ:
        case GateType::YCX:
        case GateType::YCY:
        case GateType::YCZ:
        case GateType::SWAP:
        case GateType::ISWAP:
        case GateType::ISWAP_DAG:
        case GateType::CXSWAP:
        case GateType::SWAPCX:
        case GateType::CZSWAP:
        case GateType::SQRT_XX:
        case GateType::SQRT_XX_DAG:
        case GateType::SQRT_YY:
        case GateType::SQRT_YY_DAG:
        case GateType::SQRT_ZZ:
        case GateType::SQRT_ZZ_DAG:
            kind = Kind::TWO_QUBIT;
            return true;
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
            kind = Kind::MEASURE;
            return true;
        case GateType::R:
        case GateType::RX:
        case GateType::RY:
            kind = Kind::RESET;
            return true;
        case GateType::MPAD:
            kind = Kind::MPAD;
            return true;
        case GateType::DEPOLARIZE2:
            kind = Kind::DEPOLARIZE2;
            return true;
        case GateType::PAULI_CHANNEL_2:
            kind = Kind::PAULI_CHANNEL_2;
            return true;
        default:  // MPP, SPP, pair measurements, correlated and heralded errors, ...
            return false;
    }
}

}  // namespace

Program compile(const stim::Circuit &circuit, bool reload, const uint8_t *reference, const ForcedLosses &forced_losses) {
    Program prog;
    prog.circuit = circuit.flattened();
    prog.stats = prog.circuit.compute_stats();
    prog.reload = reload;
    prog.forced_losses = forced_losses;
    const auto &ops = prog.circuit.operations;
    for (const auto &[k, qubits] : forced_losses) {
        if (k < -1 || k >= (int64_t)ops.size()) {
            throw std::invalid_argument("forced loss after instruction " + std::to_string(k) +
                                        ", but the flattened circuit has " + std::to_string(ops.size()) + " instructions");
        }
        for (uint32_t q : qubits) {
            if (q >= prog.stats.num_qubits) {
                throw std::invalid_argument("forced loss of qubit " + std::to_string(q) + ", outside the circuit");
            }
        }
    }

    // Validate, classify, and remember which measurements each detector / observable reads.
    std::vector<std::vector<uint64_t>> detector_records, observable_records(prog.stats.num_observables);
    uint64_t m = 0;
    bool has_pauli_gates = false;
    prog.kinds.reserve(ops.size());
    for (size_t k = 0; k < ops.size(); k++) {
        const auto &inst = ops[k];
        Kind kind;
        if (!classify(inst, kind)) {
            reject(k, inst, "`" + std::string(stim::GATE_DATA[inst.gate_type].name) + "` is not supported by the masked frame");
        }
        if (kind == Kind::LOSS && inst.args.size() != 1) {
            reject(k, inst, "`I_ERROR[loss]` takes exactly one probability");
        }
        if (kind == Kind::RELOAD && inst.args.size() > 1) {
            reject(k, inst, "`I_ERROR[reload]` takes at most one probability");
        }
        const bool annotation = inst.gate_type == GateType::DETECTOR || inst.gate_type == GateType::OBSERVABLE_INCLUDE;
        std::vector<uint64_t> records;
        for (const auto &t : inst.targets) {
            if (annotation && t.is_measurement_record_target()) {
                records.push_back(m - (t.data & stim::TARGET_VALUE_MASK));
            } else if (annotation) {
                reject(k, inst, "observables with Pauli targets are not supported");
            } else if (t.is_measurement_record_target() || t.is_sweep_bit_target()) {
                reject(k, inst, "classically controlled operations are not supported");
            }
        }
        if (inst.gate_type == GateType::DETECTOR) {
            detector_records.push_back(std::move(records));
        } else if (inst.gate_type == GateType::OBSERVABLE_INCLUDE) {
            auto &dst = observable_records[(size_t)inst.args[0]];
            dst.insert(dst.end(), records.begin(), records.end());
        }
        m += inst.count_measurement_results();
        has_pauli_gates |= is_pauli_gate(inst.gate_type);
        prog.kinds.push_back(kind);
    }

    prog.reference.assign(m, 0);
    if (reference != nullptr) {
        for (uint64_t i = 0; i < m; i++) {
            prog.reference[i] = reference[i] != 0;
        }
    } else {
        stim::Circuit stripped;  // loss, reload and Pauli gates removed; Pauli gates live in the frame instead
        for (const auto &inst : ops) {
            if (!is_loss(inst) && !is_reload(inst) && !is_pauli_gate(inst.gate_type)) {
                stripped.safe_append(inst);
            }
        }
        const auto bits = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(stripped);
        for (uint64_t i = 0; i < m; i++) {
            prog.reference[i] = bits[i];
        }
    }

    // Detection events are relative to Stim's reference sample, which keeps Pauli gates.  With no Pauli
    // gates and the default reference the two coincide and every constant is 0.
    prog.detector_flip.assign(detector_records.size(), 0);
    prog.observable_flip.assign(observable_records.size(), 0);
    if (reference != nullptr || has_pauli_gates) {
        const auto stim_ref = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(prog.circuit);
        auto parity = [&](const std::vector<uint64_t> &records) {
            uint8_t p = 0;
            for (uint64_t r : records) {
                p ^= prog.reference[r] ^ (uint8_t)stim_ref[r];
            }
            return p;
        };
        for (size_t d = 0; d < detector_records.size(); d++) {
            prog.detector_flip[d] = parity(detector_records[d]);
        }
        for (size_t o = 0; o < observable_records.size(); o++) {
            prog.observable_flip[o] = parity(observable_records[o]);
        }
    }
    return prog;
}

}  // namespace maskframe
