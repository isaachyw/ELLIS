# ellis._core_<variant> per SIMD variant, and ellis._cpu (no machine flags), which tells ellis/_core.py which
# variant this CPU runs.  NO_EXTRAS: no LTO and no stripping (keeps symbols for perf).
foreach(v IN LISTS ELLIS_SIMD_VARIANTS)
  pybind11_add_module(_core_${v} MODULE NO_EXTRAS bindings.cc)
  target_compile_definitions(_core_${v} PRIVATE ELLIS_MODULE_NAME=_core_${v})
  target_link_libraries(_core_${v} PRIVATE ellis_core_${v})
  target_compile_options(_core_${v} PRIVATE -O3 -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter
                         ${ELLIS_SIMD_FLAGS_${v}})
  install(TARGETS _core_${v} LIBRARY DESTINATION ellis)
endforeach()

pybind11_add_module(_cpu MODULE cpu.cc)
install(TARGETS _cpu LIBRARY DESTINATION ellis)

# Every module contains a compiled copy of Stim (Apache-2.0), so the wheel carries its license.
install(FILES ${stim_SOURCE_DIR}/LICENSE DESTINATION ellis/licenses RENAME Stim-LICENSE)
