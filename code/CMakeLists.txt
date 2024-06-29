cmake_minimum_required(VERSION 3.16)

project(cppcon2024 VERSION 1.0)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 03597a01ee50ed33e9dfd640b249b4be3799d395)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

find_package(benchmark REQUIRED)

add_library(sample INTERFACE)
target_include_directories(sample INTERFACE .)
target_compile_features(sample INTERFACE cxx_std_23)
target_compile_options(sample INTERFACE -Wall -Wconversion -Werror -Wextra -Wpedantic -O3 -DNDEBUG -fconcepts-diagnostics-depth=3)

add_executable(sample_benchmark)
target_sources(sample_benchmark PRIVATE sample_benchmark.cpp)
target_link_libraries(sample_benchmark PRIVATE sample benchmark::benchmark_main GTest::gtest)
target_compile_options(sample_benchmark PRIVATE -O3 -DNDEBUG)

add_executable(sample_test)
target_sources(sample_test PRIVATE sample_test.cpp)
target_link_libraries(sample_test PRIVATE sample GTest::gtest_main)
