# No LTO: it inlined the masked operations into one huge dispatch function (6-12% slower on bb144).
add_library(ellis_core STATIC frame/program.cc frame/sampler.cc)
target_include_directories(ellis_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(ellis_core SYSTEM PUBLIC ${stim_SOURCE_DIR}/src)  # no warnings from Stim's headers
target_link_libraries(ellis_core PUBLIC libstim)
target_compile_options(ellis_core PRIVATE -O3 -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter ${MACHINE_FLAG})
