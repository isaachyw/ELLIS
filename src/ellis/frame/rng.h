// Seeds: every batch of a sampling call gets its own std::mt19937_64 seed (Stim's generator), derived
// from the call seed and the batch index, so results do not depend on the thread count.
#pragma once

#include <cstdint>

namespace maskframe {

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t batch_seed(uint64_t call_seed, uint64_t index) {
    uint64_t s = call_seed ^ (0xD1B54A32D192ED03ULL * (index + 1));
    return splitmix64(s);
}

}  // namespace maskframe
