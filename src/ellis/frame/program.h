// A Stim circuit prepared for the masked frame: flattened, validated, and each instruction tagged with
// how the masked frame executes it (unchanged by stim::FrameSimulator, or masked).
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "stim/circuit/circuit.h"

namespace maskframe {

enum class Kind : uint8_t {
    SKIP,             // annotations, identities, I_ERROR without the `loss` or `reload` tag
    STIM,             // stim::FrameSimulator::do_gate unchanged: single-qubit Cliffords and noise, resets,
                      // detectors, observables.  A lost qubit's frame is never read again, so these need no mask.
    PAULI,            // X / Y / Z gates flip the frame (the reference is computed without them)
    TWO_QUBIT,        // two-qubit Clifford, dropped where either operand is lost
    MEASURE,          // M, MX, MY, MR, MRX, MRY: a lost qubit reads as a heralded uniformly random bit
    MPAD,
    DEPOLARIZE2,      // dropped where either operand is lost, like the gate it follows
    PAULI_CHANNEL_2,  // likewise
    LOSS,             // I_ERROR[loss](p): each target is lost with probability p
    RELOAD,           // I_ERROR[reload](p): each empty target slot gets a fresh atom in |0> with probability p (default 1)
};

using ForcedLosses = std::map<int64_t, std::vector<uint32_t>>;

struct Program {
    stim::Circuit circuit;           // flattened (forced-loss indices refer to its instructions)
    stim::CircuitStats stats;
    std::vector<Kind> kinds;         // one per instruction of `circuit`
    ForcedLosses forced_losses;      // qubits lost in every shot right after instruction k (k = -1: from the start)
    std::vector<uint8_t> reference;  // the reference record (0/1 per measurement)
    std::vector<uint8_t> detector_flip;    // XORed into detectors / observables, so that detection events keep
    std::vector<uint8_t> observable_flip;  // Stim's convention (relative to Stim's own reference sample)
};

/// Flattens and validates `circuit`.  `reference` (num_measurements bytes, or null) overrides the default
/// reference: Stim's reference sample of the circuit without loss instructions and without Pauli gates.
Program compile(const stim::Circuit &circuit, const uint8_t *reference, const ForcedLosses &forced_losses);

}  // namespace maskframe
