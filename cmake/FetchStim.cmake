# Stim, pinned to the version of the `stim` Python package.  To build against a local checkout
# instead, pass -DFETCHCONTENT_SOURCE_DIR_STIM=/path/to/Stim.  Only libstim is built (Stim's
# CMakeLists also defines its CLI, perf and test targets); it reads SIMD_WIDTH from this scope.
include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)  # keep FetchContent_Populate + add_subdirectory(EXCLUDE_FROM_ALL)
endif()
FetchContent_Declare(
  stim
  URL https://github.com/quantumlib/Stim/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH SHA256=be54b3efa6f01c16a119e0db99fe1d624f448efd5ab8219cd3da2b75a900682e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
  FetchContent_Populate(stim)
  add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
target_compile_options(libstim PRIVATE -g0 -w -fvisibility=hidden -fvisibility-inlines-hidden)
