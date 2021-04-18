cmake_minimum_required(VERSION 3.12)

project(morpheus-benchmarks)

message(STATUS "Benchmarks included")

if(NOT Morpheus_BUILD_BENCHMARKS)
    message(FATAL_ERROR "Morpheus: Morpheus_BUILD_BENCHMARKS not set.")
endif()

find_package(benchmark QUIET)
if(benchmark_DIR)
    message(STATUS "GOOGLE BENCHMARK DIR FOUND AT ${benchmark_DIR}")
endif()

if(benchmark_ROOT)
    message(STATUS "GOOGLE BENCHMARK ROOT FOUND AT ${benchmark_ROOT}")
endif()
# morpheus_configure_googlebenchmark()
