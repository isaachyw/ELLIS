// Sampling driver: batches of shots, optionally on several threads, written into row-major buffers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ellis/frame/program.h"

namespace maskframe {

/// Row-major (shots x bits) output buffers; null = not requested.  Bool rows hold one byte per bit;
/// bit-packed rows hold ceil(bits / 8) bytes in little-endian bit order (Stim's bit_packed=True layout).
struct Output {
    uint8_t *main = nullptr;         // measurements, or detectors in detector mode
    uint8_t *observables = nullptr;  // detector mode only
    uint8_t *heralds = nullptr;      // per measurement: 1 = the measured qubit was lost
    bool bit_packed = false;
};

/// SIMD widths compiled into this module, widest first.
std::vector<size_t> simd_widths();

/// The batch size used when none is given (see sampler.cc).
size_t default_batch_size(const Program &prog, bool detector_mode, bool heralds, size_t simd_width);

/// Masked-frame samples of `prog`.  Results depend on (seed, batch, simd_width) but not on `threads`.
void sample(const Program &prog, size_t shots, size_t batch, size_t simd_width, unsigned threads, uint64_t seed,
            bool detector_mode, const Output &out);

/// stim::FrameSimulator (ignores loss) run through the same batching and output code, as a speed baseline.
void stim_sample(const stim::Circuit &circuit, const std::vector<uint8_t> &reference, size_t shots, size_t batch,
                 size_t simd_width, unsigned threads, uint64_t seed, bool detector_mode, const Output &out);

}  // namespace maskframe
