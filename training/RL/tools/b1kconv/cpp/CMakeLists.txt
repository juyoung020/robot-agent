cmake_minimum_required(VERSION 3.22)
# RASC v1 C++ reader test. Data: run `b1kconv convert --out ${RASC_DIR}` first.
project(rasc_reader LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(DEFINED ENV{RA_B1K_SCENES})
  set(_rasc_def "$ENV{RA_B1K_SCENES}")
else()
  set(_rasc_def "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../data/b1k_scenes")   # config/paths.env RA_B1K_SCENES
endif()
set(RASC_DIR "${_rasc_def}" CACHE PATH "directory with <scene>.rasc + <scene>.expect")

add_executable(rasc_test rasc_test.cpp)
target_compile_options(rasc_test PRIVATE -Wall -Wextra -Wno-unused-variable)

enable_testing()
file(GLOB RASC_FILES "${RASC_DIR}/*.rasc")
add_test(NAME rasc_roundtrip_invariants COMMAND rasc_test ${RASC_FILES})
