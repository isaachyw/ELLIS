// ellis._cpu: the fastest ellis._core variant this CPU runs.  Compiled without machine flags, so it runs anywhere.
// __builtin_cpu_supports("avx2") also checks that the OS saves the AVX registers.
#include <pybind11/pybind11.h>

#include <string>

static std::string best_variant() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") ? "avx2" : "sse2";
#else
    return "polyfill";
#endif
}

PYBIND11_MODULE(_cpu, m) {
    m.def("best_variant", &best_variant, "'avx2', 'sse2' or 'polyfill': the fastest ellis._core variant this CPU runs.");
}
