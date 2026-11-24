# Stim, pinned to the version of the `stim` Python package.  To build against a local checkout instead, pass
# -DFETCHCONTENT_SOURCE_DIR_STIM=/path/to/Stim.  Stim's own CMakeLists is not used: its libstim has one SIMD width,
# and ELLIS compiles Stim's sources once per SIMD variant (src/ellis/CMakeLists.txt).
include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)  # FetchContent_Populate: download only, no add_subdirectory
endif()
FetchContent_Declare(
  stim
  URL https://github.com/quantumlib/Stim/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH SHA256=be54b3efa6f01c16a119e0db99fe1d624f448efd5ab8219cd3da2b75a900682e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
  FetchContent_Populate(stim)
endif()
file(STRINGS ${stim_SOURCE_DIR}/file_lists/source_files_no_main STIM_SOURCES)
list(TRANSFORM STIM_SOURCES PREPEND ${stim_SOURCE_DIR}/)
