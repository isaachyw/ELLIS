# CMake uses $CXX or else `c++`; on Perlmutter both mean the system GCC 7 (uv's build frontend sets
# CXX=c++), which has no C++20.  If that compiler is older than GCC 10, take `g++` from PATH (the
# gcc-native module) when it is new enough.  An explicit -DCMAKE_CXX_COMPILER always wins.
# Include this before project().
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(_cxx "$ENV{CXX}")
  if(NOT _cxx)
    set(_cxx c++)
  endif()
  separate_arguments(_cxx UNIX_COMMAND "${_cxx}")
  list(GET _cxx 0 _cxx)
  execute_process(COMMAND "${_cxx}" -dumpversion OUTPUT_VARIABLE _cxx_version
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  find_program(ELLIS_GXX NAMES g++)
  if(_cxx_version AND _cxx_version VERSION_LESS 10 AND ELLIS_GXX)
    execute_process(COMMAND "${ELLIS_GXX}" -dumpversion OUTPUT_VARIABLE _gxx_version
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_gxx_version VERSION_GREATER_EQUAL 10)
      message(STATUS "${_cxx} is GCC ${_cxx_version} (no C++20); using ${ELLIS_GXX} (GCC ${_gxx_version})")
      set(CMAKE_CXX_COMPILER "${ELLIS_GXX}")
    endif()
  endif()
endif()
