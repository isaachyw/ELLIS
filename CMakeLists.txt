# ELLIS: the C++ core (src/ellis) and its Python module ellis._core (src/python), built on Stim's C++ core.
#
#   pip/uv:  `uv sync` builds this through scikit-build-core (see pyproject.toml).
#   by hand: cmake -S . -B build/dev -G Ninja && cmake --build build/dev -j8
cmake_minimum_required(VERSION 3.18)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(SelectCompiler)
project(ellis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # every module keeps its own copy of Stim to itself
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(SimdVariants)
include(FetchStim)

add_subdirectory(src/ellis)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(src/python)
