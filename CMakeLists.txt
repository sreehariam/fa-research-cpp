cmake_minimum_required(VERSION 3.20)
project(fa_research_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA kernels" OFF)

# Library sources
add_library(fa_cpu
    src/cpu/tensor.cpp
    src/common/checks.cpp
    src/common/math.cpp
    src/common/random.cpp
    src/attention_ref.cpp
    src/mask.cpp
)

target_include_directories(fa_cpu PUBLIC include)
target_compile_features(fa_cpu PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(fa_cpu PRIVATE /W4 /permissive-)
else()
  target_compile_options(fa_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Testing via FetchContent (self-contained)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(unit_tests
  tests/test_tensor.cpp
  tests/test_utils.cpp
)
target_link_libraries(unit_tests PRIVATE fa_cpu gtest gtest_main)
add_test(NAME all_tests COMMAND unit_tests)

message(STATUS "USE_CUDA=${USE_CUDA} (CUDA not used in base commit)")
