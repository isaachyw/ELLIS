// MSC S-proxy reference: prefix tableau, phase check, CSS gauge sweep and terminal queries,
// with the full tableau as the exact fallback (paper section 6).
#pragma once

#include "ellis/reference/full_tableau.h"

#include <algorithm>
#include <bit>
#include <tuple>

namespace ellis::reference {

// XZ convention: i^phase X^x Z^z. Stabilizer rows commute, but their
// multiplication still needs the crossing phase from moving Z past X.
struct Row {
    uint64_t x = 0;
    uint64_t z = 0;
    uint8_t phase = 0;
};

inline void multiply(Row &a, const Row &b) {
    a.phase = (a.phase + b.phase + 2 * std::popcount(a.z & b.x)) & 3;
    a.x ^= b.x;
    a.z ^= b.z;
}

using QueryInput = std::tuple<size_t, std::vector<uint32_t>, std::string, uint32_t, bool>;

struct Query {
    size_t record;
    std::vector<uint32_t> qubits;
    std::string bases;
    uint32_t pivot;
    bool inverted;
};

class FastPlan : public Plan {
    int64_t cut;
    int64_t boundary;
    std::vector<uint32_t> core;
    std::vector<int> core_index;
    std::vector<Query> queries;
    size_t num_records = 0;
    bool supported = true;

    static bool css_gate(stim::GateType gate) {
        using G = stim::GateType;
        return gate == G::CX || gate == G::SWAP || gate == G::I ||
               gate == G::R || gate == G::RX || gate == G::M ||
               gate == G::MX || gate == G::MR || gate == G::MRX;
    }

    // Eliminate all rows carrying an X or Z on a lost core qubit. The
    // remaining subgroup is exactly the stabilizer group of the live marginal.
    std::vector<Row> live_rows(const stim::TableauSimulator<64> &sim,
                               const std::vector<uint8_t> &lost) const {
        std::vector<Row> rows;
        for (const auto &pauli : sim.canonical_stabilizers()) {
            Row row;
            for (size_t i = 0; i < core.size(); ++i) {
                row.x |= uint64_t(bool(pauli.xs[i])) << i;
                row.z |= uint64_t(bool(pauli.zs[i])) << i;
            }
            row.phase = (std::popcount(row.x & row.z) + 2 * pauli.sign) & 3;
            rows.push_back(row);
        }
        size_t removed = 0;
        for (size_t q = 0; q < core.size(); ++q) {
            if (!lost[core[q]]) continue;
            for (bool z_column : {false, true}) {
                size_t pivot = removed;
                auto bit = [&](const Row &r) {
                    return ((z_column ? r.z : r.x) >> q) & 1;
                };
                while (pivot < rows.size() && !bit(rows[pivot])) ++pivot;
                if (pivot == rows.size()) continue;
                std::swap(rows[pivot], rows[removed]);
                for (size_t j = removed + 1; j < rows.size(); ++j) {
                    if (bit(rows[j])) multiply(rows[j], rows[removed]);
                }
                ++removed;
            }
        }
        rows.erase(rows.begin(), rows.begin() + removed);
        return rows;
    }

    bool find_gauge(const std::vector<Row> &rows, uint64_t &u, uint64_t &v) const {
        for (const Row &a : rows) {
            const bool ta = std::popcount(a.x & a.z) & 1;
            for (const Row &b : rows) {
                const bool tb = std::popcount(b.x & b.z) & 1;
                if (bool(std::popcount(a.z & b.x) & 1) != (ta && tb)) return false;
            }
        }
        struct Equation { uint64_t lo, hi; bool rhs; };
        std::vector<Equation> equations;
        for (const Row &r : rows) {
            const unsigned t = std::popcount(r.x & r.z) & 1;
            equations.push_back({r.z, r.x, bool(((r.phase + 4 - t) & 3) >> 1)});
        }
        std::vector<size_t> pivots;
        size_t rank = 0;
        for (size_t column = 0; column < 2 * core.size(); ++column) {
            const bool high = column >= core.size();
            const size_t q = high ? column - core.size() : column;
            auto bit = [&](const Equation &e) { return ((high ? e.hi : e.lo) >> q) & 1; };
            size_t pivot = rank;
            while (pivot < equations.size() && !bit(equations[pivot])) ++pivot;
            if (pivot == equations.size()) continue;
            std::swap(equations[pivot], equations[rank]);
            for (size_t j = 0; j < equations.size(); ++j) {
                if (j != rank && bit(equations[j])) {
                    equations[j].lo ^= equations[rank].lo;
                    equations[j].hi ^= equations[rank].hi;
                    equations[j].rhs ^= equations[rank].rhs;
                }
            }
            pivots.push_back(column);
            ++rank;
        }
        for (size_t j = rank; j < equations.size(); ++j) {
            if (equations[j].rhs) return false;
        }
        u = v = 0;
        for (size_t j = 0; j < rank; ++j) {
            if (!equations[j].rhs) continue;
            if (pivots[j] < core.size()) u |= uint64_t(1) << pivots[j];
            else v |= uint64_t(1) << (pivots[j] - core.size());
        }
        return true;
    }

  public:
    FastPlan(std::vector<Operation> ops, size_t num_qubits,
             int64_t cut, int64_t boundary, std::vector<uint32_t> core,
             const std::vector<QueryInput> &query_inputs)
        : Plan(std::move(ops), num_qubits), cut(cut), boundary(boundary),
          core(std::move(core)), core_index(num_qubits, -1) {
        supported = this->core.size() <= 64 && cut <= boundary;
        for (size_t i = 0; i < this->core.size(); ++i) {
            const uint32_t q = this->core[i];
            if (q >= num_qubits || core_index[q] >= 0) {
                throw std::invalid_argument("invalid or duplicate core qubit");
            }
            core_index[q] = i;
        }
        for (const Operation &op : operations) {
            if (op.kind == Kind::Pad || op.kind == Kind::Measurement) {
                num_records += op.targets.size();
            }
            if (op.kind == Kind::Reload && op.index >= boundary) supported = false;  // terminal queries assume no reload
            if (op.index >= boundary || op.kind == Kind::Pad) continue;
            if (op.index >= cut && !css_gate(op.gate)) supported = false;
            const size_t stride = op.kind == Kind::Two ? 2 : 1;
            for (size_t start = 0; start < op.targets.size(); start += stride) {
                bool any_core = false, all_core = true;
                for (size_t j = 0; j < stride; ++j) {
                    const bool in_core = core_index[op.targets[start + j].qubit_value()] >= 0;
                    any_core |= in_core;
                    all_core &= in_core;
                }
                if (op.index < cut && ((any_core && !all_core) ||
                                      (!any_core && !css_gate(op.gate)))) supported = false;
            }
        }
        for (const auto &[record, qubits, bases, pivot, inverted] : query_inputs) {
            if (record >= num_records || pivot >= num_qubits || qubits.size() != bases.size()) {
                throw std::invalid_argument("invalid terminal query");
            }
            for (size_t j = 0; j < qubits.size(); ++j) {
                if (qubits[j] >= num_qubits || (bases[j] != 'X' && bases[j] != 'Y' && bases[j] != 'Z')) {
                    throw std::invalid_argument("invalid terminal Pauli factor");
                }
            }
            queries.push_back({record, qubits, bases, pivot, inverted});
        }
    }

    std::pair<std::string, bool> reference_fast(const Losses &losses) const {
        validate_losses(losses);
        auto fallback = [&]() { return std::make_pair(Plan::reference(losses), false); };
        if (!supported || (!losses.empty() && losses.rbegin()->first >= boundary)) return fallback();
        std::string record(num_records, '\0');
        std::vector<uint8_t> lost(num_qubits, 0), px(num_qubits, 0), pz(num_qubits, 0);
        auto pending = losses.begin();
        auto advance_losses = [&](int64_t index) {
            while (pending != losses.end() && pending->first < index) {
                for (uint32_t q : pending->second) lost[q] = 1;
                ++pending;
            }
        };
        stim::TableauSimulator<64> sim(std::mt19937_64(0), core.size(), +1);
        size_t slot = 0, first_suffix = 0;
        std::vector<stim::GateTarget> targets;
        for (; first_suffix < operations.size(); ++first_suffix) {
            const Operation &op = operations[first_suffix];
            if (op.index >= cut) break;
            advance_losses(op.index);
            if (op.kind == Kind::Pad) {
                for (const auto &target : op.targets) record[slot++] = target.qubit_value();
                continue;
            }
            if (op.kind == Kind::Reload) {  // a fresh |0> atom in each empty slot
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (!lost[q]) continue;
                    lost[q] = 0;
                    if (core_index[q] >= 0) {
                        const auto t = stim::GateTarget::qubit(core_index[q]);
                        sim.do_gate({stim::GateType::R, {}, &t, ""});
                    }
                }
                continue;
            }
            const size_t stride = op.kind == Kind::Two ? 2 : 1;
            for (size_t start = 0; start < op.targets.size(); start += stride) {
                const auto &target = op.targets[start];
                bool present = true;
                targets.clear();
                for (size_t j = 0; j < stride; ++j) {
                    const auto &t = op.targets[start + j];
                    const uint32_t qt = t.qubit_value();
                    present &= !lost[qt];
                    if (core_index[qt] >= 0) {
                        targets.push_back(stim::GateTarget::qubit(core_index[qt], t.is_inverted_result_target()));
                    }
                }
                if (op.kind == Kind::Measurement && !present) {
                    ++slot; // A lost readout is the zero reference coin.
                } else if (present) {
                    if (!targets.empty()) sim.do_gate({op.gate, {}, targets, ""});
                    if (op.kind == Kind::Measurement) {
                        record[slot++] = targets.empty() ? target.is_inverted_result_target()
                                                        : bool(sim.measurement_record.storage.back());
                    }
                }
            }
        }
        advance_losses(cut);
        uint64_t u, v;
        if (!find_gauge(live_rows(sim, lost), u, v)) return fallback();
        for (size_t i = 0; i < core.size(); ++i) {
            if (!lost[core[i]]) {
                px[core[i]] = (u >> i) & 1;
                pz[core[i]] = (v >> i) & 1;
            }
        }
        using G = stim::GateType;
        for (size_t k = first_suffix; k < operations.size(); ++k) {
            const Operation &op = operations[k];
            if (op.index >= boundary) break;
            advance_losses(op.index);
            if (op.kind == Kind::Pad) {
                for (const auto &target : op.targets) record[slot++] = target.qubit_value();
                continue;
            }
            if (op.kind == Kind::Reload) {  // a fresh |0> atom: no frame Pauli, as after a reset
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (lost[q]) {
                        lost[q] = 0;
                        px[q] = pz[q] = 0;
                    }
                }
                continue;
            }
            if (op.kind == Kind::Measurement) {
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (!lost[q]) {
                        const bool x_measure = op.gate == G::MX || op.gate == G::MRX;
                        record[slot] = (x_measure ? pz[q] : px[q]) ^ target.is_inverted_result_target();
                    }
                    ++slot;
                    if (op.measure_reset && !lost[q]) px[q] = pz[q] = 0;
                }
            } else if (op.reset) {
                for (const auto &target : op.targets) {
                    const uint32_t q = target.qubit_value();
                    if (!lost[q]) px[q] = pz[q] = 0;
                }
            } else if (op.gate == G::CX || op.gate == G::SWAP) {
                for (size_t j = 0; j < op.targets.size(); j += 2) {
                    const uint32_t a = op.targets[j].qubit_value(), b = op.targets[j + 1].qubit_value();
                    if (lost[a] || lost[b]) continue;
                    if (op.gate == G::CX) {
                        px[b] ^= px[a];
                        pz[a] ^= pz[b];
                    } else {
                        std::swap(px[a], px[b]);
                        std::swap(pz[a], pz[b]);
                    }
                }
            }
        }
        advance_losses(boundary);
        struct EffectiveQuery { bool coin; bool mixed; unsigned ys; bool pure_x; };
        std::vector<EffectiveQuery> effective;
        effective.reserve(queries.size());
        std::vector<uint8_t> mixed_axes(num_qubits, 0);
        size_t mixed_count = 0;
        for (const Query &query : queries) {
            EffectiveQuery e{bool(lost[query.pivot]), false, 0, true};
            bool pure_x = true, pure_z = true;
            if (!e.coin) {
                for (size_t j = 0; j < query.qubits.size(); ++j) {
                    if (lost[query.qubits[j]]) continue;
                    const char basis = query.bases[j];
                    pure_x &= basis == 'X';
                    pure_z &= basis == 'Z';
                    e.ys += basis == 'Y';
                }
            }
            e.pure_x = pure_x;
            e.mixed = !e.coin && !pure_x && !pure_z;
            if (e.mixed) {
                if (++mixed_count > 1) return fallback();
                for (size_t j = 0; j < query.qubits.size(); ++j) {
                    const uint32_t q = query.qubits[j];
                    if (!lost[q]) {
                        const char basis = query.bases[j];
                        mixed_axes[q] = basis == 'X' ? 1 : basis == 'Z' ? 2 : 3;
                    }
                }
            }
            effective.push_back(e);
        }
        for (size_t i = 0; i < effective.size(); ++i) {
            if (!effective[i].mixed) continue;
            using Support = std::vector<uint64_t>;
            const size_t words = (num_qubits + 63) / 64;
            std::vector<Support> effect_x, effect_z;
            auto dot = [&](const Support &a, const Support &b) {
                unsigned parity = 0;
                for (size_t w = 0; w < words; ++w) parity ^= std::popcount(a[w] & b[w]) & 1;
                return bool(parity);
            };
            // Backward sandwiches P E P produce the CSS projector E = K^dag K
            // of all later zero-result queries. A new X check removes one
            // anticommuting Z direction, and conversely for a new Z check.
            for (size_t j = queries.size(); j-- > i + 1;) {
                if (effective[j].coin) continue;
                const Query &later = queries[j];
                Support p(words, 0);
                bool nonempty = false;
                for (uint32_t q : later.qubits) {
                    if (!lost[q]) {
                        p[q / 64] |= uint64_t(1) << (q % 64);
                        nonempty = true;
                    }
                }
                if (!nonempty) continue;
                auto &opposite = effective[j].pure_x ? effect_z : effect_x;
                size_t pivot = 0;
                while (pivot < opposite.size() && !dot(opposite[pivot], p)) ++pivot;
                if (pivot < opposite.size()) {
                    for (size_t k = 0; k < opposite.size(); ++k) {
                        if (k != pivot && dot(opposite[k], p)) {
                            for (size_t w = 0; w < words; ++w) opposite[k][w] ^= opposite[pivot][w];
                        }
                    }
                    opposite.erase(opposite.begin() + pivot);
                }
                auto &same = effective[j].pure_x ? effect_x : effect_z;
                same.push_back(std::move(p));
            }
            Support qx(words, 0), qz(words, 0);
            for (size_t q = 0; q < num_qubits; ++q) {
                if (mixed_axes[q] & 1) qx[q / 64] |= uint64_t(1) << (q % 64);
                if (mixed_axes[q] & 2) qz[q / 64] |= uint64_t(1) << (q % 64);
            }
            bool anti_x = false, anti_z = false;
            for (const Support &x : effect_x) anti_x |= dot(x, qz);
            for (const Support &z : effect_z) anti_z |= dot(z, qx);
            // Commuting with either whole sector is sufficient; individual
            // later queries can anticommute after their effects cancel.
            if (anti_x && anti_z) return fallback();
        }
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto &query = queries[i];
            const auto &e = effective[i];
            if (e.coin) continue;
            bool value = ((e.ys / 2) & 1) ^ query.inverted;
            for (size_t j = 0; j < query.qubits.size(); ++j) {
                const uint32_t q = query.qubits[j];
                if (lost[q]) continue;
                const char basis = query.bases[j];
                value ^= (basis != 'X' && px[q]) ^ (basis != 'Z' && pz[q]);
            }
            record[query.record] = value;
        }
        return {std::move(record), true};
    }
};

}  // namespace ellis::reference
