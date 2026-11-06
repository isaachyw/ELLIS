# NO_EXTRAS: no LTO and no stripping (keeps symbols for perf).
pybind11_add_module(_core MODULE NO_EXTRAS bindings.cc)
target_link_libraries(_core PRIVATE ellis_core)
target_compile_options(_core PRIVATE -O3 -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter ${MACHINE_FLAG})
install(TARGETS _core LIBRARY DESTINATION ellis)
