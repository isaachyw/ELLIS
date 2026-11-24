// The masked Pauli frame: stim::FrameSimulator plus a per-shot `alive` bit per qubit.
//
// Losing a qubit clears its alive bit and nothing reads its frame again.  Two-qubit gates (and their
// DEPOLARIZE2 / PAULI_CHANNEL_2 noise) are dropped where either operand is lost, a lost qubit's readout
// is a heralded uniformly random placeholder bit, and `I_ERROR[reload]` puts a fresh |0> atom into an empty
// slot.  Everything else is stim::FrameSimulator's own code, unchanged: single-qubit gates and noise,
// resets, detectors and observables, measurement-flip noise, the record, and the random stream.  With
// nothing lost and no reload marker, the samples are bit-for-bit stim::FrameSimulator's for the same seed (a
// marker draws its fresh gauge bits in every shot, so the random stream does not depend on which shots lost).
#pragma once

#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "ellis/frame/program.h"
#include "ellis/frame/rng.h"
#include "stim/gates/gates.h"
#include "stim/simulators/frame_simulator.h"
#include "stim/util_bot/probability_util.h"

#define MASKFRAME_INLINE inline __attribute__((always_inline))

namespace maskframe {

template <size_t W>
struct MaskedFrameSimulator : stim::FrameSimulator<W> {
    using Base = stim::FrameSimulator<W>;
    using Word = stim::bitword<W>;

    const Program &prog;
    const bool keep_heralds;
    const unsigned log_batch;              // the batch size is a power of two
    stim::simd_bit_table<W> alive_table;   // alive_table[q][s]: qubit q is present in shot s
    stim::simd_bit_table<W> herald_table;  // herald_table[i][s]: measurement i of shot s read a lost qubit
    stim::simd_bits<W> scratch;            // 4 rows, for PAULI_CHANNEL_2 and reload
    uint64_t num_results = 0;

    MaskedFrameSimulator(const Program &prog, size_t batch, bool keep_heralds)
        : Base(prog.stats, stim::FrameSimulatorMode::STORE_EVERYTHING_TO_MEMORY, batch, std::mt19937_64(0)),
          prog(prog),
          keep_heralds(keep_heralds),
          log_batch((unsigned)std::countr_zero(batch)),
          alive_table(prog.stats.num_qubits, batch),
          herald_table(keep_heralds ? prog.stats.num_measurements : 0, batch),
          scratch(4 * batch) {
    }

    /// Simulates one batch from |0...0> with every qubit present.
    void run(uint64_t seed) {
        this->rng.seed(seed);
        this->reset_all();
        std::memset(alive_table.data.u8, 0xFF, alive_table.data.num_u8_padded());
        num_results = 0;
        const auto &ops = prog.circuit.operations;
        for (size_t k = 0; k < ops.size(); k++) {
            execute(ops[k], prog.kinds[k]);
        }
    }

   private:
    MASKFRAME_INLINE size_t words() const {
        return this->x_table.num_simd_words_minor;
    }
    MASKFRAME_INLINE void set_present(size_t q) {
        auto row = alive_table[q];
        std::memset(row.u8, 0xFF, row.num_u8_padded());
    }

    void execute(const stim::CircuitInstruction &inst, Kind kind) {
        using stim::GateType;
        switch (kind) {
            case Kind::SKIP:
                return;
            case Kind::STIM:
                this->do_gate(inst);
                return;
            case Kind::PAULI:
                for (auto t : inst.targets) {
                    if (inst.gate_type != GateType::Z) {
                        this->x_table[t.data].invert_bits();
                    }
                    if (inst.gate_type != GateType::X) {
                        this->z_table[t.data].invert_bits();
                    }
                }
                return;
            case Kind::TWO_QUBIT:
                two_qubit(inst);
                return;
            case Kind::MEASURE:
                switch (inst.gate_type) {
                    case GateType::MX: measure<0, false>(inst); break;
                    case GateType::MY: measure<1, false>(inst); break;
                    case GateType::M: measure<2, false>(inst); break;
                    case GateType::MRX: measure<0, true>(inst); break;
                    case GateType::MRY: measure<1, true>(inst); break;
                    default: measure<2, true>(inst); break;
                }
                return;
            case Kind::MPAD:
                this->do_MPAD(inst);
                for (size_t k = 0; k < inst.targets.size(); k++, num_results++) {
                    if (keep_heralds) {
                        herald_table[num_results].clear();
                    }
                }
                return;
            case Kind::DEPOLARIZE2:
                depolarize2(inst);
                return;
            case Kind::PAULI_CHANNEL_2:
                pauli_channel_2(inst);
                return;
            case Kind::LOSS:
                loss(inst);
                return;
            case Kind::RELOAD:
                reload(inst);
                return;
        }
    }

    /// Applies body(x1, z1, x2, z2, m) to every SIMD word of every target pair, m = both operands present.
    template <typename BODY>
    MASKFRAME_INLINE void masked_pairs(const stim::CircuitInstruction &inst, BODY body) {
        const size_t n = words();
        const auto &t = inst.targets;
        for (size_t k = 0; k < t.size(); k += 2) {
            const size_t q1 = t[k].data, q2 = t[k + 1].data;
            Word *__restrict x1 = this->x_table[q1].ptr_simd;
            Word *__restrict z1 = this->z_table[q1].ptr_simd;
            Word *__restrict x2 = this->x_table[q2].ptr_simd;
            Word *__restrict z2 = this->z_table[q2].ptr_simd;
            const Word *__restrict a1 = alive_table[q1].ptr_simd;
            const Word *__restrict a2 = alive_table[q2].ptr_simd;
            for (size_t w = 0; w < n; w++) {
                body(x1[w], z1[w], x2[w], z2[w], a1[w] & a2[w]);
            }
        }
    }

    /// stim::FrameSimulator's two-qubit frame updates, each written as deltas of the old values AND-ed with m.
    void two_qubit(const stim::CircuitInstruction &inst) {
        using G = stim::GateType;
        switch (inst.gate_type) {
            case G::CX:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    z1 ^= z2 & m;
                    x2 ^= x1 & m;
                });
            case G::CY:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d1 = (x2 ^ z2) & m, d2 = x1 & m;
                    z1 ^= d1;
                    z2 ^= d2;
                    x2 ^= d2;
                });
            case G::CZ:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d1 = x2 & m, d2 = x1 & m;
                    z1 ^= d1;
                    z2 ^= d2;
                });
            case G::XCX:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d1 = z2 & m, d2 = z1 & m;
                    x1 ^= d1;
                    x2 ^= d2;
                });
            case G::XCY:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d1 = (x2 ^ z2) & m, d2 = z1 & m;
                    x1 ^= d1;
                    x2 ^= d2;
                    z2 ^= d2;
                });
            case G::XCZ:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d1 = x2 & m, d2 = z1 & m;
                    x1 ^= d1;
                    z2 ^= d2;
                });
            case G::YCX:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d2 = (x1 ^ z1) & m, d1 = z2 & m;
                    x2 ^= d2;
                    x1 ^= d1;
                    z1 ^= d1;
                });
            case G::YCY:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word y1 = (x1 ^ z1) & m, y2 = (x2 ^ z2) & m;
                    x1 ^= y2;
                    z1 ^= y2;
                    x2 ^= y1;
                    z2 ^= y1;
                });
            case G::YCZ:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d2 = (x1 ^ z1) & m, d1 = x2 & m;
                    z2 ^= d2;
                    z1 ^= d1;
                    x1 ^= d1;
                });
            case G::SWAP:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dx = (x1 ^ x2) & m, dz = (z1 ^ z2) & m;
                    x1 ^= dx;
                    x2 ^= dx;
                    z1 ^= dz;
                    z2 ^= dz;
                });
            case G::ISWAP:
            case G::ISWAP_DAG:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dx = (x1 ^ x2) & m, dz = (x1 ^ x2 ^ z1 ^ z2) & m;
                    x1 ^= dx;
                    x2 ^= dx;
                    z1 ^= dz;
                    z2 ^= dz;
                });
            case G::CXSWAP:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dz1 = (z1 ^ z2) & m, dz2 = z1 & m, dx1 = x2 & m, dx2 = (x1 ^ x2) & m;
                    z1 ^= dz1;
                    z2 ^= dz2;
                    x1 ^= dx1;
                    x2 ^= dx2;
                });
            case G::SWAPCX:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dz1 = z2 & m, dz2 = (z1 ^ z2) & m, dx1 = (x1 ^ x2) & m, dx2 = x1 & m;
                    z1 ^= dz1;
                    z2 ^= dz2;
                    x1 ^= dx1;
                    x2 ^= dx2;
                });
            case G::CZSWAP:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word s = z1 ^ z2, dx = (x1 ^ x2) & m, dz1 = (s ^ x1) & m, dz2 = (s ^ x2) & m;
                    z1 ^= dz1;
                    z2 ^= dz2;
                    x1 ^= dx;
                    x2 ^= dx;
                });
            case G::SQRT_XX:
            case G::SQRT_XX_DAG:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dz = (z1 ^ z2) & m;
                    x1 ^= dz;
                    x2 ^= dz;
                });
            case G::SQRT_YY:
            case G::SQRT_YY_DAG:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word d = (x1 ^ z1 ^ x2 ^ z2) & m;
                    x1 ^= d;
                    z1 ^= d;
                    x2 ^= d;
                    z2 ^= d;
                });
            case G::SQRT_ZZ:
            case G::SQRT_ZZ_DAG:
                return masked_pairs(inst, [](Word &x1, Word &z1, Word &x2, Word &z2, Word m) {
                    const Word dx = (x1 ^ x2) & m;
                    z1 ^= dx;
                    z2 ^= dx;
                });
            default:
                throw std::logic_error("maskframe: not a two-qubit Clifford");
        }
    }

    /// stim::FrameSimulator::do_M{X,Y,Z} / do_MR{X,Y,Z}, step for step and with the same random stream,
    /// except that the recorded row is masked: a lost shot records the collapse randomization instead,
    /// a uniformly random placeholder bit (its frame is never read again).  BASIS: 0 = X, 1 = Y, 2 = Z.
    template <int BASIS, bool RESET>
    void measure(const stim::CircuitInstruction &inst) {
        auto &record = this->m_record;
        record.reserve_noisy_space_for_results(inst, this->rng);  // measurement-flip noise
        const size_t n = words();
        for (auto t : inst.targets) {
            const size_t q = t.qubit_value();
            if constexpr (BASIS == 1) {
                this->x_table[q] ^= this->z_table[q];
            }
            auto randomized = BASIS == 0 ? this->x_table[q] : this->z_table[q];
            randomized.randomize(randomized.num_bits_padded(), this->rng);
            const Word *__restrict v = (BASIS == 0 ? this->z_table[q] : this->x_table[q]).ptr_simd;
            const Word *__restrict r = randomized.ptr_simd;
            const Word *__restrict a = alive_table[q].ptr_simd;
            Word *__restrict out = record.storage[record.stored].ptr_simd;
            Word *__restrict h = keep_heralds ? herald_table[num_results].ptr_simd : nullptr;
            for (size_t w = 0; w < n; w++) {
                out[w] ^= (v[w] & a[w]) | a[w].andnot(r[w]);
                if (h != nullptr) {
                    h[w] = ~a[w];
                }
            }
            record.stored++;
            record.unwritten++;
            num_results++;
            if constexpr (!RESET) {
                if constexpr (BASIS == 1) {
                    this->x_table[q] ^= this->z_table[q];
                }
            } else {
                if constexpr (BASIS == 2) {
                    this->x_table[q].clear();
                } else if constexpr (BASIS == 0) {
                    this->z_table[q].clear();
                } else {
                    this->x_table[q] = this->z_table[q];
                }
            }
        }
    }

    /// I_ERROR[reload](p): in each shot where the slot is empty, with probability p (default 1), a fresh |0>
    /// atom: X part 0, Z part a fresh random gauge bit (as Stim's reset), present again.  A failed reload
    /// leaves the slot empty; present atoms are untouched.  The random bits are drawn in every shot, so the
    /// random stream does not depend on losses.
    void reload(const stim::CircuitInstruction &inst) {
        const size_t n = words();
        const float p = inst.args.empty() ? 1.0f : (float)inst.args[0];
        auto fresh = scratch.word_range_ref(0, n);
        auto success = scratch.word_range_ref(n, n);
        for (auto t : inst.targets) {
            const size_t q = t.data;
            fresh.randomize(fresh.num_bits_padded(), this->rng);
            if (p < 1) {
                stim::biased_randomize_bits(p, success.u64, success.u64 + success.num_u64_padded(), this->rng);
            }
            Word *x = this->x_table[q].ptr_simd, *z = this->z_table[q].ptr_simd, *a = alive_table[q].ptr_simd;
            for (size_t w = 0; w < n; w++) {
                const Word filled = p < 1 ? a[w].andnot(success.ptr_simd[w]) : ~a[w];
                x[w] = filled.andnot(x[w]);
                z[w] = filled.andnot(z[w]) | (filled & fresh.ptr_simd[w]);
                a[w] |= filled;
            }
        }
    }

    /// stim::FrameSimulator::do_DEPOLARIZE2, skipping pairs with a lost operand.  The Pauli is drawn
    /// before the check, so the random stream does not depend on losses.
    void depolarize2(const stim::CircuitInstruction &inst) {
        const auto &targets = inst.targets;
        const size_t batch_size = this->batch_size;
        stim::RareErrorIterator::for_samples(inst.args[0], (targets.size() * batch_size) >> 1, this->rng, [&](size_t s) {
            const auto p = 1 + (this->rng() % 15);
            const size_t target_index = (s >> log_batch) << 1;  // batch sizes are powers of two
            const size_t sample_index = s & (batch_size - 1);
            const size_t t1 = targets[target_index].data, t2 = targets[target_index + 1].data;
            if (!(alive_table[t1][sample_index] && alive_table[t2][sample_index])) {
                return;
            }
            this->x_table[t1][sample_index] ^= (bool)(p & 1);
            this->z_table[t1][sample_index] ^= (bool)(p & 2);
            this->x_table[t2][sample_index] ^= (bool)(p & 4);
            this->z_table[t2][sample_index] ^= (bool)(p & 8);
        });
    }

    /// stim::FrameSimulator::do_PAULI_CHANNEL_2 on each pair, undone where an operand is lost.
    void pauli_channel_2(const stim::CircuitInstruction &inst) {
        const size_t n = words();
        Word *saved = scratch.ptr_simd;
        for (size_t k = 0; k < inst.targets.size(); k += 2) {
            const size_t q1 = inst.targets[k].data, q2 = inst.targets[k + 1].data;
            Word *rows[4] = {this->x_table[q1].ptr_simd, this->z_table[q1].ptr_simd, this->x_table[q2].ptr_simd,
                             this->z_table[q2].ptr_simd};
            for (size_t j = 0; j < 4; j++) {
                std::copy(rows[j], rows[j] + n, saved + j * n);
            }
            const stim::GateTarget *pair = inst.targets.ptr_start + k;
            Base::do_PAULI_CHANNEL_2(stim::CircuitInstruction(inst.gate_type, inst.args, {pair, pair + 2}, inst.tag));
            const Word *a1 = alive_table[q1].ptr_simd, *a2 = alive_table[q2].ptr_simd;
            for (size_t j = 0; j < 4; j++) {
                for (size_t w = 0; w < n; w++) {
                    const Word m = a1[w] & a2[w];
                    rows[j][w] = (rows[j][w] & m) | m.andnot(saved[j * n + w]);
                }
            }
        }
    }

    /// Uniform integer in [0, n), exactly (Lemire's nearly divisionless rejection).
    MASKFRAME_INLINE uint64_t uniform_below(uint64_t n) {
        unsigned __int128 m = (unsigned __int128)this->rng() * n;
        if ((uint64_t)m < n) {
            const uint64_t threshold = (uint64_t)(-n) % n;
            while ((uint64_t)m < threshold) {
                m = (unsigned __int128)this->rng() * n;
            }
        }
        return (uint64_t)(m >> 64);
    }

    /// I_ERROR[loss](p): each (target, shot) is lost with probability p.  Losing is idempotent, so instead
    /// of a Bernoulli trial per (target, shot) this drops Poisson(n λ) uniformly random hits, λ = -ln(1 - p):
    /// each (target, shot) is then hit at least once, independently, with probability 1 - e^-λ = p exactly,
    /// and a hit costs one random integer instead of a logarithm (the cost of a geometric gap).
    void loss(const stim::CircuitInstruction &inst) {
        const double p = inst.args[0];
        const auto &targets = inst.targets;
        if (!(p > 0)) {
            return;
        }
        if (p >= 1) {
            for (auto t : targets) {
                alive_table[t.data].clear();
            }
            return;
        }
        const uint64_t n = (uint64_t)targets.size() << log_batch;
        std::poisson_distribution<uint64_t> hits(-std::log1p(-p) * (double)n);
        uint64_t k = hits(this->rng);
        const uint64_t shot_mask = this->batch_size - 1;
        uint64_t *alive = alive_table.data.u64;
        const size_t row_u64 = alive_table.num_simd_words_minor * (W / 64);
        auto lose = [&](uint64_t s) {
            const uint64_t shot = s & shot_mask;
            alive[targets[s >> log_batch].data * row_u64 + (shot >> 6)] &= ~(uint64_t{1} << (shot & 63));
        };
        if (n > UINT32_MAX) {
            for (; k > 0; k--) {
                lose(uniform_below(n));
            }
            return;
        }
        // Two exact 32-bit draws (Lemire's rejection) per 64-bit number from a splitmix64 stream seeded by
        // Stim's generator: 14-19% faster than one mt19937_64 draw per hit at p_loss = 5e-3 (bb144 and
        // surface d = 25), where hits are the hot loop.
        const uint32_t n32 = (uint32_t)n, threshold = (uint32_t)(0u - n32) % n32;
        uint64_t state = this->rng();
        auto draw = [&](uint32_t r) {
            const uint64_t m = (uint64_t)r * n32;
            if ((uint32_t)m >= threshold) {
                lose(m >> 32);
                k--;
            }
        };
        while (k > 0) {
            const uint64_t r = splitmix64(state);
            draw((uint32_t)r);
            if (k > 0) {
                draw((uint32_t)(r >> 32));
            }
        }
    }
};

}  // namespace maskframe
