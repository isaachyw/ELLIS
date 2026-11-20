// Python bindings, ellis._core: the masked-frame sampler (MaskFrameSampler, StimSampler) and the exact
// per-loss-pattern references of ellis.msc (Plan, FastPlan).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <string>

#include "ellis/frame/program.h"
#include "ellis/frame/rng.h"
#include "ellis/frame/sampler.h"
#include "ellis/reference/full_tableau.h"
#include "ellis/reference/prefix.h"
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

    MaskFrameSampler(const py::object &circuit, const py::object &reference,
                     const py::object &forced_losses, const py::object &seed, size_t batch_size, size_t simd_width)
        : config(batch_size, simd_width, seed) {
        const stim::Circuit parsed(py::str(circuit).cast<std::string>());
        maskframe::ForcedLosses losses;
        if (!forced_losses.is_none()) {
            for (const auto &[k, qubits] : forced_losses.cast<py::dict>()) {
                auto &dst = losses[k.cast<int64_t>()];
                for (const auto &q : qubits) {
                    dst.push_back(q.cast<uint32_t>());
                }
            }
        }
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> ref;
        if (!reference.is_none()) {
            ref = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(reference);
            if (!ref || ref.ndim() != 1 || (uint64_t)ref.size() != parsed.count_measurements()) {
                throw std::invalid_argument(
                    "reference must be a 1d array with one entry per measurement (" +
                    std::to_string(parsed.count_measurements()) + ")");
            }
        }
        prog = maskframe::compile(parsed, reference.is_none() ? nullptr : ref.data(), losses);
    }

    size_t batch(bool detector_mode, bool heralds) const {
        return config.batch_size ? config.batch_size
                                 : maskframe::default_batch_size(prog, detector_mode, heralds, config.simd_width);
    }

    py::object sample(size_t shots, bool bit_packed, bool heralds, unsigned threads) {
        py::array meas = bit_array(shots, prog.stats.num_measurements, bit_packed);
        py::array her = bit_array(heralds ? shots : 0, prog.stats.num_measurements, bit_packed);
        maskframe::Output out{data(meas), nullptr, heralds ? data(her) : nullptr, bit_packed};
        const uint64_t seed = config.next_seed();
        const size_t b = batch(false, heralds);
        {
            py::gil_scoped_release release;
            maskframe::sample(prog, shots, b, config.simd_width, threads, seed, false, out);
        }
        if (heralds) {
            return py::make_tuple(meas, her);
        }
        return meas;
    }

    py::object sample_detectors(size_t shots, bool bit_packed, bool heralds, unsigned threads) {
        py::array dets = bit_array(shots, prog.stats.num_detectors, bit_packed);
        py::array obs = bit_array(shots, prog.stats.num_observables, bit_packed);
        py::array her = bit_array(heralds ? shots : 0, prog.stats.num_measurements, bit_packed);
        maskframe::Output out{data(dets), data(obs), heralds ? data(her) : nullptr, bit_packed};
        const uint64_t seed = config.next_seed();
        const size_t b = batch(true, heralds);
        {
            py::gil_scoped_release release;
            maskframe::sample(prog, shots, b, config.simd_width, threads, seed, true, out);
        }
        if (heralds) {
            return py::make_tuple(dets, obs, her);
        }
        return py::make_tuple(dets, obs);
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

    py::array sample(size_t shots, bool bit_packed, unsigned threads) {
        if (reference.size() != stats.num_measurements) {
            const auto bits = stim::TableauSimulator<stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(circuit);
            reference.assign(stats.num_measurements, 0);
            for (size_t i = 0; i < reference.size(); i++) {
                reference[i] = bits[i];
            }
        }
        py::array meas = bit_array(shots, stats.num_measurements, bit_packed);
        maskframe::Output out{data(meas), nullptr, nullptr, bit_packed};
        const uint64_t seed = config.next_seed();
        {
            py::gil_scoped_release release;
            maskframe::stim_sample(circuit, reference, shots, batch(), config.simd_width, threads, seed, false, out);
        }
        return meas;
    }

    py::tuple sample_detectors(size_t shots, bool bit_packed, unsigned threads) {
        py::array dets = bit_array(shots, stats.num_detectors, bit_packed);
        py::array obs = bit_array(shots, stats.num_observables, bit_packed);
        maskframe::Output out{data(dets), data(obs), nullptr, bit_packed};
        const uint64_t seed = config.next_seed();
        {
            py::gil_scoped_release release;
            maskframe::stim_sample(circuit, reference, shots, batch(), config.simd_width, threads, seed, true, out);
        }
        return py::make_tuple(dets, obs);
    }
};

// frame.reference_skeleton(): (index, kind, gate name, targets); a readout target is (qubit, inverted).
std::vector<ellis::reference::Operation> parse_skeleton(const py::iterable &skeleton, size_t num_qubits) {
    std::vector<ellis::reference::Operation> ops;
    for (const py::handle item : skeleton) {
        const auto entry = py::reinterpret_borrow<py::tuple>(item);
        const auto kind = entry[1].cast<std::string>();
        std::vector<std::pair<uint32_t, bool>> targets;
        for (const py::handle target : entry[3]) {
            if (kind == "meas") {
                const auto pair = py::reinterpret_borrow<py::tuple>(target);
                targets.emplace_back(pair[0].cast<uint32_t>(), pair[1].cast<bool>());
            } else {
                targets.emplace_back(py::cast<uint32_t>(target), false);
            }
        }
        ops.push_back(ellis::reference::make_operation(entry[0].cast<int64_t>(), kind, entry[2].cast<std::string>(),
                                                       targets, num_qubits));
    }
    return ops;
}

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
    forced_losses: {k: [qubits]} lost in every shot right after instruction k of
        circuit.flattened() (k = -1: lost from the start), in addition to random loss.
    seed: seed of the sampler's random stream (None: from the OS).  Each sample call continues it.
    batch_size: shots simulated together per thread (a power of two >= the SIMD width); 0 = automatic.
    simd_width: 64, 128 or 256 bits; 0 = the widest compiled in.
)DOC")
        .def(py::init<const py::object &, const py::object &, const py::object &, const py::object &, size_t,
                      size_t>(),
             "circuit"_a, py::kw_only(), "reference"_a = py::none(), "forced_losses"_a = py::none(),
             "seed"_a = py::none(), "batch_size"_a = 0, "simd_width"_a = 0)
        .def("sample", &MaskFrameSampler::sample, "shots"_a, py::kw_only(), "bit_packed"_a = false,
             "heralds"_a = true, "threads"_a = 1,
             R"DOC(Samples measurement records.

Returns (measurements, heralds), each (shots, num_measurements): bool arrays, or uint8 arrays of
little-endian packed bits if bit_packed.  heralds[s, i] is 1 when measurement i of shot s read a lost
qubit.  With heralds=False only the measurements are returned.)DOC")
        .def("sample_detectors", &MaskFrameSampler::sample_detectors, "shots"_a, py::kw_only(),
             "bit_packed"_a = false, "heralds"_a = false, "threads"_a = 1,
             R"DOC(Samples detection events (Stim's convention) and observable flips.

Returns (detectors, observables) or, with heralds=True, (detectors, observables, heralds) where
heralds are per measurement as in sample().)DOC")
        .def_property_readonly("reference", &MaskFrameSampler::reference)
        .def_property_readonly("num_qubits", [](const MaskFrameSampler &s) { return s.prog.stats.num_qubits; })
        .def_property_readonly("num_measurements", [](const MaskFrameSampler &s) { return s.prog.stats.num_measurements; })
        .def_property_readonly("num_detectors", [](const MaskFrameSampler &s) { return s.prog.stats.num_detectors; })
        .def_property_readonly("num_observables", [](const MaskFrameSampler &s) { return s.prog.stats.num_observables; })
        .def_property_readonly("num_instructions", [](const MaskFrameSampler &s) { return s.prog.circuit.operations.size(); })
        .def_property_readonly("simd_width", [](const MaskFrameSampler &s) { return s.config.simd_width; })
        .def(
            "batch_size",
            [](const MaskFrameSampler &s, bool detectors, bool heralds) { return s.batch(detectors, heralds); },
            py::kw_only(), "detectors"_a = false, "heralds"_a = true,
            "The batch size used by sample (detectors=False) or sample_detectors (detectors=True).");

    py::class_<StimSampler>(m, "StimSampler", R"DOC(
Stim's own stim::FrameSimulator (ignores loss, no masks), compiled into this module with the same flags
and run through the same batching, threading and output code as MaskFrameSampler: the speed baseline.
)DOC")
        .def(py::init<const py::object &, const py::object &, size_t, size_t>(), "circuit"_a, py::kw_only(),
             "seed"_a = py::none(), "batch_size"_a = 0, "simd_width"_a = 0)
        .def("sample", &StimSampler::sample, "shots"_a, py::kw_only(), "bit_packed"_a = false, "threads"_a = 1)
        .def("sample_detectors", &StimSampler::sample_detectors, "shots"_a, py::kw_only(), "bit_packed"_a = false,
             "threads"_a = 1)
        .def_property_readonly("simd_width", [](const StimSampler &s) { return s.config.simd_width; })
        .def_property_readonly("batch_size", &StimSampler::batch);

    using ellis::reference::FastPlan;
    using ellis::reference::Losses;
    using ellis::reference::Plan;
    using ellis::reference::QueryInput;

    py::class_<Plan>(m, "Plan")
        .def(py::init([](const py::iterable &skeleton, size_t num_qubits) {
                 return Plan(parse_skeleton(skeleton, num_qubits), num_qubits);
             }),
             py::arg("skeleton"), py::arg("num_qubits"))
        .def("reference", [](const Plan &plan, const Losses &losses) {
            std::string record;
            {
                py::gil_scoped_release release;
                record = plan.reference(losses);
            }
            return py::bytes(record);
        })
        .def("batch_reference", [](const Plan &plan, const std::vector<Losses> &patterns) {
            std::vector<std::string> records;
            records.reserve(patterns.size());
            {
                py::gil_scoped_release release;
                for (const Losses &losses : patterns) {
                    records.push_back(plan.reference(losses));
                }
            }
            py::list result;
            for (const std::string &record : records) {
                result.append(py::bytes(record));
            }
            return result;
        });
    py::class_<FastPlan>(m, "FastPlan")
        .def(py::init([](const py::iterable &skeleton, size_t num_qubits, int64_t cut, int64_t boundary,
                         std::vector<uint32_t> core, const std::vector<QueryInput> &queries) {
                 return FastPlan(parse_skeleton(skeleton, num_qubits), num_qubits, cut, boundary,
                                 std::move(core), queries);
             }),
             py::arg("skeleton"), py::arg("num_qubits"),
             py::arg("cut"), py::arg("boundary"), py::arg("core_qubits"), py::arg("queries"))
        .def("reference", [](const FastPlan &plan, const Losses &losses) {
            std::pair<std::string, bool> result;
            {
                py::gil_scoped_release release;
                result = plan.reference_fast(losses);
            }
            return py::make_tuple(py::bytes(result.first), result.second);
        })
        .def("batch_reference", [](const FastPlan &plan, const std::vector<Losses> &patterns) {
            std::vector<std::pair<std::string, bool>> records;
            records.reserve(patterns.size());
            {
                py::gil_scoped_release release;
                for (const Losses &losses : patterns) records.push_back(plan.reference_fast(losses));
            }
            py::list result;
            for (const auto &record : records) result.append(py::make_tuple(py::bytes(record.first), record.second));
            return result;
        });
}
