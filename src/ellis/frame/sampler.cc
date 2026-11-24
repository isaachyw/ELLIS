#include "ellis/frame/sampler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "ellis/frame/rng.h"
#include "ellis/frame/simulator.h"
#include "stim/simulators/frame_simulator.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace maskframe {
namespace {

/// dst[i] = bit i of src, for i < n.
void expand_bits(const uint8_t *src, uint8_t *dst, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256i spread = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i select = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i one = _mm256_set1_epi8(1);
    for (; i + 32 <= n; i += 32) {
        uint32_t v;
        std::memcpy(&v, src + (i >> 3), sizeof(v));
        __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)v), spread);
        b = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, select), select), one);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), b);
    }
#endif
    static const std::array<uint64_t, 256> SPREAD = [] {
        std::array<uint64_t, 256> table{};
        for (size_t b = 0; b < 256; b++) {
            for (size_t k = 0; k < 8; k++) {
                table[b] |= (uint64_t)((b >> k) & 1) << (8 * k);
            }
        }
        return table;
    }();
    for (; i + 8 <= n; i += 8) {
        std::memcpy(dst + i, &SPREAD[src[i >> 3]], 8);
    }
    for (; i < n; i++) {
        dst[i] = (src[i >> 3] >> (i & 7)) & 1;
    }
}

/// Writes a batch table (row r = bit r of every shot; `nw` words per row) into row-major (shots x bits)
/// output, one W x W bit block at a time with Stim's in-place transpose kernel.  Rows with flip[r] set
/// are inverted on the way (the reference record, detector constants).
template <size_t W>
struct RowWriter {
    using Word = stim::bitword<W>;
    stim::simd_bits<W> block{W * W};

    void write(const Word *rows, size_t nw, size_t num_rows, const uint8_t *flip, uint8_t *out, bool packed,
               size_t first_shot, size_t num_shots) {
        if (num_rows == 0 || out == nullptr) {
            return;
        }
        Word *blk = block.ptr_simd;
        const Word ones = ~Word();
        const size_t stride = packed ? (num_rows + 7) / 8 : num_rows;
        for (size_t s = 0; s * W < num_shots; s++) {
            const size_t shots_here = std::min(W, num_shots - s * W);
            for (size_t r0 = 0; r0 < num_rows; r0 += W) {
                const size_t rows_here = std::min(W, num_rows - r0);
                for (size_t k = 0; k < rows_here; k++) {
                    blk[k] = rows[(r0 + k) * nw + s];
                    if (flip != nullptr && flip[r0 + k]) {
                        blk[k] ^= ones;
                    }
                }
                for (size_t k = rows_here; k < W; k++) {
                    blk[k] = Word();
                }
                Word::inplace_transpose_square(blk, 1);  // now blk[j] = bits r0.. of shot s * W + j
                for (size_t j = 0; j < shots_here; j++) {
                    uint8_t *dst = out + (first_shot + s * W + j) * stride;
                    const uint8_t *src = reinterpret_cast<const uint8_t *>(&blk[j]);
                    if (packed) {
                        std::memcpy(dst + r0 / 8, src, std::min(W / 8, stride - r0 / 8));
                    } else {
                        expand_bits(src, dst + r0, rows_here);
                    }
                }
            }
        }
    }
};

/// Runs body(claim) on `threads` threads; claim() hands out batch indices, then returns SIZE_MAX.
template <typename BODY>
void run_batches(unsigned threads, size_t num_batches, BODY &&body) {
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto claim = [&]() -> size_t {
        const size_t b = next.fetch_add(1, std::memory_order_relaxed);
        return b < num_batches ? b : SIZE_MAX;
    };
    auto loop = [&]() {
        try {
            body(claim);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = std::current_exception();
            }
            next.store(num_batches);
        }
    };
    const size_t n = std::clamp<size_t>(threads, 1, std::max<size_t>(num_batches, 1));
    if (n == 1) {
        loop();
    } else {
        std::vector<std::thread> pool;
        for (size_t k = 0; k < n; k++) {
            pool.emplace_back(loop);
        }
        for (auto &t : pool) {
            t.join();
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

template <size_t W>
void sample_w(const Program &prog, size_t shots, size_t batch, unsigned threads, uint64_t seed, const Output &out) {
    const auto &stats = prog.stats;
    run_batches(threads, (shots + batch - 1) / batch, [&](auto &claim) {
        MaskedFrameSimulator<W> sim(prog, batch, out.heralds != nullptr);
        RowWriter<W> writer;
        for (size_t b; (b = claim()) != SIZE_MAX;) {
            sim.run(batch_seed(seed, b));
            const size_t first = b * batch, count = std::min(batch, shots - first);
            const auto &meas = sim.m_record.storage;
            const auto &dets = sim.det_record.storage;
            const auto &her = sim.herald_table;
            writer.write(meas.data.ptr_simd, meas.num_simd_words_minor, stats.num_measurements, prog.reference.data(),
                         out.measurements, out.bit_packed, first, count);
            writer.write(dets.data.ptr_simd, dets.num_simd_words_minor, stats.num_detectors, prog.detector_flip.data(),
                         out.detectors, out.bit_packed, first, count);
            writer.write(sim.obs_record.data.ptr_simd, sim.obs_record.num_simd_words_minor, stats.num_observables,
                         prog.observable_flip.data(), out.observables, out.bit_packed, first, count);
            writer.write(her.data.ptr_simd, her.num_simd_words_minor, stats.num_measurements, nullptr, out.heralds,
                         out.bit_packed, first, count);
        }
    });
}

template <size_t W>
void stim_sample_w(const stim::Circuit &circuit, const std::vector<uint8_t> &reference, size_t shots, size_t batch,
                   unsigned threads, uint64_t seed, const Output &out) {
    const stim::CircuitStats stats = circuit.compute_stats();
    run_batches(threads, (shots + batch - 1) / batch, [&](auto &claim) {
        stim::FrameSimulator<W> sim(stats, stim::FrameSimulatorMode::STORE_EVERYTHING_TO_MEMORY, batch,
                                    std::mt19937_64(0));
        RowWriter<W> writer;
        for (size_t b; (b = claim()) != SIZE_MAX;) {
            sim.rng.seed(batch_seed(seed, b));
            sim.reset_all();
            circuit.for_each_operation([&](const stim::CircuitInstruction &op) { sim.do_gate(op); });
            const size_t first = b * batch, count = std::min(batch, shots - first);
            const auto &meas = sim.m_record.storage;
            const auto &dets = sim.det_record.storage;
            writer.write(meas.data.ptr_simd, meas.num_simd_words_minor, stats.num_measurements, reference.data(),
                         out.measurements, out.bit_packed, first, count);
            writer.write(dets.data.ptr_simd, dets.num_simd_words_minor, stats.num_detectors, nullptr, out.detectors,
                         out.bit_packed, first, count);
            writer.write(sim.obs_record.data.ptr_simd, sim.obs_record.num_simd_words_minor, stats.num_observables,
                         nullptr, out.observables, out.bit_packed, first, count);
        }
    });
}

void check_batch(size_t batch, size_t simd_width) {
    if (batch < simd_width || (batch & (batch - 1)) != 0) {
        throw std::invalid_argument(
            "batch_size must be a power of two and at least the SIMD width (" + std::to_string(simd_width) + ")");
    }
}

}  // namespace

std::vector<size_t> simd_widths() {
    std::vector<size_t> out;
#if defined(__AVX2__)
    out.push_back(256);
#endif
#if defined(__SSE2__)
    out.push_back(128);
#endif
    out.push_back(64);
    return out;
}

size_t default_batch_size(const Program &prog, size_t simd_width) {
    // The largest batch, up to 2048 shots, whose frame (x, z and alive rows) fits in a 512 KB L2: loss hits
    // and noise are random accesses into it, and records, detectors and heralds are streamed.  Measured on
    // EPYC 7763 for bb72, bb144 and surface codes d = 11..31 (best or within 5% of the best batch).
    const uint64_t frame_rows = 3 * (uint64_t)prog.stats.num_qubits;
    constexpr uint64_t L2_BYTES = 512 << 10;
    size_t batch = 2048;
    while (batch > simd_width && frame_rows * batch / 8 > L2_BYTES) {
        batch >>= 1;
    }
    return std::max(batch, simd_width);
}

void sample(const Program &prog, size_t shots, size_t batch, size_t simd_width, unsigned threads, uint64_t seed,
            const Output &out) {
    check_batch(batch, simd_width);
    switch (simd_width) {
#if defined(__AVX2__)
        case 256:
            return sample_w<256>(prog, shots, batch, threads, seed, out);
#endif
#if defined(__SSE2__)
        case 128:
            return sample_w<128>(prog, shots, batch, threads, seed, out);
#endif
        case 64:
            return sample_w<64>(prog, shots, batch, threads, seed, out);
        default:
            throw std::invalid_argument("unsupported simd_width " + std::to_string(simd_width));
    }
}

void stim_sample(const stim::Circuit &circuit, const std::vector<uint8_t> &reference, size_t shots, size_t batch,
                 size_t simd_width, unsigned threads, uint64_t seed, const Output &out) {
    check_batch(batch, simd_width);
    switch (simd_width) {
#if defined(__AVX2__)
        case 256:
            return stim_sample_w<256>(circuit, reference, shots, batch, threads, seed, out);
#endif
#if defined(__SSE2__)
        case 128:
            return stim_sample_w<128>(circuit, reference, shots, batch, threads, seed, out);
#endif
        case 64:
            return stim_sample_w<64>(circuit, reference, shots, batch, threads, seed, out);
        default:
            throw std::invalid_argument("unsupported simd_width " + std::to_string(simd_width));
    }
}

}  // namespace maskframe
