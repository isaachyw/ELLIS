# The SIMD variants of ellis._core to build, each its own module ellis._core_<variant>.  Stim fixes its SIMD word
# width at compile time from the machine flags (__AVX2__: 256 bits, __SSE2__: 128, else 64), so every variant
# compiles its own copy of Stim with its own flags, and ellis/_core.py imports the fastest variant this CPU runs.
#
#   ELLIS_SIMD=native (default)  the one variant for the building machine: a source build runs where it was built
#   ELLIS_SIMD="avx2;sse2"       the x86-64 wheels (pyproject.toml, [tool.cibuildwheel])
#
# x86-64 always has SSE2 (it is in the ABI), so polyfill (64-bit words) is the build for every other architecture.
set(ELLIS_SIMD native CACHE STRING "SIMD variants of ellis._core: native, or a list of avx2, sse2, polyfill")

set(ELLIS_SIMD_FLAGS_avx2 -mavx2 -msse2)
set(ELLIS_SIMD_FLAGS_sse2 -msse2 -mno-avx2)
set(ELLIS_SIMD_FLAGS_polyfill "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_supported avx2 sse2)
else()
  set(_supported polyfill)
endif()

if(ELLIS_SIMD STREQUAL "native")
  if(_supported STREQUAL "polyfill")
    set(ELLIS_SIMD_VARIANTS polyfill)
  else()
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("int main() { return __builtin_cpu_supports(\"avx2\") ? 0 : 1; }" ELLIS_HOST_HAS_AVX2)
    if(ELLIS_HOST_HAS_AVX2)
      set(ELLIS_SIMD_VARIANTS avx2)
    else()
      set(ELLIS_SIMD_VARIANTS sse2)
    endif()
  endif()
else()
  set(ELLIS_SIMD_VARIANTS ${ELLIS_SIMD})
endif()

foreach(_v IN LISTS ELLIS_SIMD_VARIANTS)
  if(NOT _v IN_LIST _supported)
    message(FATAL_ERROR "ELLIS_SIMD variant '${_v}' cannot be built for ${CMAKE_SYSTEM_PROCESSOR}; choose from: ${_supported}")
  endif()
endforeach()
message(STATUS "ellis._core SIMD variants: ${ELLIS_SIMD_VARIANTS}")
