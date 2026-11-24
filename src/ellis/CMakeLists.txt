# One Stim and one ELLIS core per SIMD variant (cmake/SimdVariants.cmake), all compiled with the variant's flags.
# No LTO: it inlined the masked operations into one huge dispatch function (6-12% slower on bb144).
foreach(v IN LISTS ELLIS_SIMD_VARIANTS)
  add_library(stim_${v} STATIC ${STIM_SOURCES})
  target_include_directories(stim_${v} SYSTEM PUBLIC ${stim_SOURCE_DIR}/src)  # no warnings from Stim's headers
  target_compile_options(stim_${v} PRIVATE -O3 -g0 -w -fno-strict-aliasing ${ELLIS_SIMD_FLAGS_${v}})

  add_library(ellis_core_${v} STATIC frame/program.cc frame/sampler.cc)
  target_include_directories(ellis_core_${v} PUBLIC ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(ellis_core_${v} PUBLIC stim_${v})
  target_compile_options(ellis_core_${v} PRIVATE -O3 -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter
                         ${ELLIS_SIMD_FLAGS_${v}})
endforeach()
