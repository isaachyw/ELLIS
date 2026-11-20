// Exact noiseless reference of one loss pattern: the full tableau of the surviving circuit.
#pragma once

#include "stim/circuit/circuit_instruction.h"
#include "stim/gates/gates.h"
#include "stim/simulators/tableau_simulator.h"

#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ellis::reference {

using Losses = std::map<int64_t, std::vector<uint32_t>>;

enum class Kind { One, Two, Measurement, Pad, Reload };

struct Operation {
    int64_t index;
    Kind kind;
    stim::GateType gate;
    std::vector<stim::GateTarget> targets;
    bool reset;
    bool measure_reset;
};

// One entry of frame.reference_skeleton: kind is "one", "two", "meas", "pad" or "reload" (an
// `I_ERROR[reload]` marker, name "R"); a target is (qubit, inverted), and only readouts may be inverted.
inline Operation make_operation(int64_t index, const std::string &kind_name, const std::string &name,
                                const std::vector<std::pair<uint32_t, bool>> &targets, size_t num_qubits) {
    const auto kind = kind_name == "pad"      ? Kind::Pad
                      : kind_name == "meas"   ? Kind::Measurement
                      : kind_name == "two"    ? Kind::Two
                      : kind_name == "reload" ? Kind::Reload
                                               : Kind::One;
    const bool reset = kind == Kind::One && (name == "R" || name == "RX" || name == "RY");
    const bool measure_reset = name == "MR" || name == "MRX" || name == "MRY";
    Operation op{index, kind, stim::GATE_DATA.at(name).id, {}, reset, measure_reset};
    for (const auto &[q, inverted] : targets) {
        if (kind != Kind::Pad && q >= num_qubits) {
            throw std::invalid_argument("instruction target is outside the circuit");
        }
        op.targets.push_back(stim::GateTarget::qubit(q, kind == Kind::Measurement && inverted));
    }
    return op;
}

class Plan {
  protected:
    std::vector<Operation> operations;
    size_t num_qubits;

    void validate_losses(const Losses &losses) const {
        for (const auto &[index, qubits] : losses) {
            (void)index;
            for (uint32_t q : qubits) {
                if (q >= num_qubits) {
                    throw std::invalid_argument("loss target is outside the circuit");
                }
            }
        }
    }

  public:
    Plan(std::vector<Operation> ops, size_t num_qubits) : operations(std::move(ops)), num_qubits(num_qubits) {}

    std::string reference(const Losses &losses) const {
        validate_losses(losses);
        stim::TableauSimulator<128> sim(std::mt19937_64(0), num_qubits, +1);
        std::vector<uint8_t> lost(num_qubits, 0);
        std::vector<stim::GateTarget> kept;
        auto pending = losses.begin();
        for (const Operation &op : operations) {
            // Losses occur after the original instruction, including index -1.
            while (pending != losses.end() && pending->first < op.index) {
                for (uint32_t q : pending->second) {
                    lost[q] = 1;
                }
                ++pending;
            }
            if (op.kind == Kind::Pad) {
                sim.do_gate({op.gate, {}, op.targets, ""});
                continue;
            }
            if (op.kind == Kind::Reload) {  // a fresh |0> atom in each empty slot
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (lost[q]) {
                        lost[q] = 0;
                        sim.do_gate({stim::GateType::R, {}, &target, ""});
                    }
                }
                continue;
            }
            if (op.kind == Kind::Measurement) {
                kept.clear();
                auto flush_measurements = [&]() {
                    if (!kept.empty()) {
                        sim.do_gate({op.gate, {}, kept, ""});
                        kept.clear();
                    }
                };
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (!lost[q]) {
                        kept.push_back(target);
                    } else {
                        flush_measurements();
                        sim.measurement_record.record_result(false);
                    }
                }
                flush_measurements();
                continue;
            }
            kept.clear();
            const size_t stride = op.kind == Kind::Two ? 2 : 1;
            for (size_t start = 0; start < op.targets.size(); start += stride) {
                bool present = true;
                for (size_t j = 0; j < stride; ++j) {
                    present &= !lost[op.targets[start + j].qubit_value()];
                }
                if (present) {
                    for (size_t j = 0; j < stride; ++j) {
                        kept.push_back(op.targets[start + j]);
                    }
                }
            }
            if (!kept.empty()) {
                sim.do_gate({op.gate, {}, kept, ""});
            }
        }
        const auto &record = sim.measurement_record.storage;
        std::string result(record.size(), '\0');
        for (size_t i = 0; i < record.size(); ++i) {
            result[i] = record[i] ? '\1' : '\0';
        }
        return result;
    }
};


}  // namespace ellis::reference
