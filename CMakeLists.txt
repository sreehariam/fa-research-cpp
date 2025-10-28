cmake_minimum_required(VERSION 3.20)
project(fa_research_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA kernels" OFF)

# --------------------------
# Library sources (CPU base)
# --------------------------
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
if (MSVC)
  target_compile_options(fa_cpu PRIVATE /W4 /permissive-)
else()
  target_compile_options(fa_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --------------------------
# Testing setup (GoogleTest)
# --------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  # Optional: silence CMP0135 warning by forcing extraction timestamp
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Auto-collect all test files under tests/ (so new tests are picked up automatically)
file(GLOB TEST_SRCS CONFIGURE_DEPENDS tests/*.cpp)

add_executable(unit_tests ${TEST_SRCS})
target_link_libraries(unit_tests PRIVATE fa_cpu gtest gtest_main)

# IMPORTANT: Discover *individual* gtest cases as separate CTest tests
include(GoogleTest)
gtest_discover_tests(unit_tests
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DISCOVERY_MODE PRE_TEST
  DISCOVERY_TIMEOUT 60
)

# If you want colored gtest output in CI logs
add_compile_definitions(GTEST_COLOR=1)

message(STATUS "USE_CUDA=${USE_CUDA} (CUDA not used in base commit)")
