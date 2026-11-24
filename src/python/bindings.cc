// Python bindings, ellis._core: the masked-frame sampler (MaskFrameSampler, StimSampler).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <string>

#include "ellis/frame/program.h"
#include "ellis/frame/rng.h"
#include "ellis/frame/sampler.h"
#include "stim/simulators/tableau_simulator.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

uint64_t initial_seed(const py::object &seed) {
    if (seed.is_none()) {
        std::random_device device;
        return ((uint64_t)device() << 32) ^ (uint64_t)device();
    }
    return (uint64_t)seed.cast<int64_t>();
}

py::array bit_array(size_t shots, size_t bits, bool packed) {
    if (packed) {
        return py::array_t<uint8_t>(std::vector<py::ssize_t>{(py::ssize_t)shots, (py::ssize_t)((bits + 7) / 8)});
    }
    return py::array_t<bool>(std::vector<py::ssize_t>{(py::ssize_t)shots, (py::ssize_t)bits});
}

uint8_t *data(py::array &a) {
    return a.size() ? static_cast<uint8_t *>(a.mutable_data()) : nullptr;
}

size_t pick_simd_width(size_t requested) {
    const auto widths = maskframe::simd_widths();
    if (requested == 0) {
        return widths.front();
    }
    for (size_t w : widths) {
        if (w == requested) {
            return w;
        }
    }
    throw std::invalid_argument("simd_width " + std::to_string(requested) + " is not compiled into this module");
}

/// Shared by both samplers: seed stream and batch size.
struct Config {
    size_t batch_size;  // 0 = per-call default
    size_t simd_width;
    uint64_t seed_state;

    Config(size_t batch_size, size_t simd_width, const py::object &seed)
        : batch_size(batch_size), simd_width(pick_simd_width(simd_width)), seed_state(initial_seed(seed)) {
    }
    uint64_t next_seed() {
        return maskframe::splitmix64(seed_state);
    }
};

struct MaskFrameSampler {
    maskframe::Program prog;
    Config config;

    MaskFrameSampler(const py::object &circuit, const py::object &reference, const py::object &seed, size_t batch_size,
                     size_t simd_width)
        : config(batch_size, simd_width, seed) {
        const stim::Circuit parsed(py::str(circuit).cast<std::string>());
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> ref;
        if (!reference.is_none()) {
            ref = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(reference);
            if (!ref || ref.ndim() != 1 || (uint64_t)ref.size() != parsed.count_measurements()) {
                throw std::invalid_argument(
                    "reference must be a 1d array with one entry per measurement (" +
                    std::to_string(parsed.count_measurements()) + ")");
            }
        }
        prog = maskframe::compile(parsed, reference.is_none() ? nullptr : ref.data());
    }

    size_t batch() const {
        return config.batch_size ? config.batch_size : maskframe::default_batch_size(prog, config.simd_width);
    }

    py::tuple sample(size_t shots, bool bit_packed, unsigned threads) {
        const auto &st = prog.stats;
        py::array meas = bit_array(shots, st.num_measurements, bit_packed);
        py::array dets = bit_array(shots, st.num_detectors, bit_packed);
        py::array obs = bit_array(shots, st.num_observables, bit_packed);
        py::array her = bit_array(shots, st.num_measurements, bit_packed);
        maskframe::Output out{data(meas), data(dets), data(obs), data(her), bit_packed};
        const uint64_t seed = config.next_seed();
        {
            py::gil_scoped_release release;
            maskframe::sample(prog, shots, batch(), config.simd_width, threads, seed, out);
        }
        return py::make_tuple(meas, dets, obs, her);
    }

    py::array reference() const {
        py::array_t<bool> out((py::ssize_t)prog.reference.size());
        std::copy(prog.reference.begin(), prog.reference.end(), reinterpret_cast<uint8_t *>(out.mutable_data()));
        return out;
    }
};

struct StimSampler {
    stim::Circuit circuit;
    stim::CircuitStats stats;
    std::vector<uint8_t> reference;
    Config config;

    StimSampler(const py::object &circuit, const py::object &seed, size_t batch_size, size_t simd_width)
        : circuit(py::str(circuit).cast<std::string>()), stats(this->circuit.compute_stats()),
          config(batch_size, simd_width, seed) {
    }

    size_t batch() const {
        return config.batch_size ? config.batch_size : 1024;
    }

    py::tuple sample(size_t shots, bool bit_packed, unsigned threads) {
        if (reference.size() != stats.num_measurements) {
            const auto bits = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(circuit);
            reference.assign(stats.num_measurements, 0);
            for (size_t i = 0; i < reference.size(); i++) {
                reference[i] = bits[i];
            }
        }
        py::array meas = bit_array(shots, stats.num_measurements, bit_packed);
        py::array dets = bit_array(shots, stats.num_detectors, bit_packed);
        py::array obs = bit_array(shots, stats.num_observables, bit_packed);
        maskframe::Output out{data(meas), data(dets), data(obs), nullptr, bit_packed};
        const uint64_t seed = config.next_seed();
        {
            py::gil_scoped_release release;
            maskframe::stim_sample(circuit, reference, shots, batch(), config.simd_width, threads, seed, out);
        }
        return py::make_tuple(meas, dets, obs);
    }
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ masked Pauli-frame sampler for Clifford circuits with atom loss (gate-drop semantics).";
    m.attr("SIMD_WIDTHS") = maskframe::simd_widths();

    py::class_<MaskFrameSampler>(m, "MaskFrameSampler", R"DOC(
Masked Pauli-frame sampler compiled from a Stim circuit with `I_ERROR[loss](p)` loss instructions.

Each shot runs the circuit minus every operation on a qubit after it is lost: two-qubit gates (and
DEPOLARIZE2 / PAULI_CHANNEL_2) are dropped when either operand is lost, a lost qubit's readouts are
heralded and replaced by uniformly random placeholder bits, and `I_ERROR[reload](p)` puts a fresh |0>
atom into an empty slot (write it before a reset to reload there).  Measurements are the frame record XOR one reference record, which is exact whenever
that reference is a valid noiseless sample of every shot's lossy circuit (see ellis.certify).

Args:
    circuit: a stim.Circuit or its text.
    reference: reference record (one bool per measurement).  Default: Stim's reference sample of
        the circuit without loss instructions and without Pauli gates (they live in the frame).
    seed: seed of the sampler's random stream (None: from the OS).  Each sample call continues it.
    batch_size: shots simulated together per thread (a power of two >= the SIMD width); 0 = automatic.
    simd_width: 64, 128 or 256 bits; 0 = the widest compiled in.
)DOC")
        .def(py::init<const py::object &, const py::object &, const py::object &, size_t, size_t>(), "circuit"_a,
             py::kw_only(), "reference"_a = py::none(), "seed"_a = py::none(), "batch_size"_a = 0, "simd_width"_a = 0)
        .def("sample", &MaskFrameSampler::sample, "shots"_a, py::kw_only(), "bit_packed"_a = false, "threads"_a = 1,
             R"DOC(Samples measurements, detection events, observable flips and heralds in one simulation pass.

Returns (measurements, detectors, observables, heralds), each (shots, n): bool arrays, or uint8 arrays of
little-endian packed bits if bit_packed.  Detection events and observable flips follow Stim's convention
(relative to Stim's noiseless reference sample).  heralds[s, i] is 1 when measurement i of shot s read a
lost qubit.)DOC")
        .def_property_readonly("reference", &MaskFrameSampler::reference)
        .def_property_readonly("num_qubits", [](const MaskFrameSampler &s) { return s.prog.stats.num_qubits; })
        .def_property_readonly("num_measurements", [](const MaskFrameSampler &s) { return s.prog.stats.num_measurements; })
        .def_property_readonly("num_detectors", [](const MaskFrameSampler &s) { return s.prog.stats.num_detectors; })
        .def_property_readonly("num_observables", [](const MaskFrameSampler &s) { return s.prog.stats.num_observables; })
        .def_property_readonly("num_instructions", [](const MaskFrameSampler &s) { return s.prog.circuit.operations.size(); })
        .def_property_readonly("simd_width", [](const MaskFrameSampler &s) { return s.config.simd_width; })
        .def_property_readonly("batch_size", &MaskFrameSampler::batch, "Shots simulated together per thread.");

    py::class_<StimSampler>(m, "StimSampler", R"DOC(
Stim's own stim::FrameSimulator (ignores loss, no masks), compiled into this module with the same flags
and run through the same batching, threading and output code as MaskFrameSampler: the speed baseline.
)DOC")
        .def(py::init<const py::object &, const py::object &, size_t, size_t>(), "circuit"_a, py::kw_only(),
             "seed"_a = py::none(), "batch_size"_a = 0, "simd_width"_a = 0)
        .def("sample", &StimSampler::sample, "shots"_a, py::kw_only(), "bit_packed"_a = false, "threads"_a = 1,
             "(measurements, detectors, observables) from one simulation pass.")
        .def_property_readonly("simd_width", [](const StimSampler &s) { return s.config.simd_width; })
        .def_property_readonly("batch_size", &StimSampler::batch);
}
