message(STATUS "Benchmarks are enabled")

# This function takes in an example name and example source and handles setting
# all of the associated properties and linking to build the example
function(ConfigureBenchmark BENCHMARK_NAME BENCHMARK_SRC)
  add_executable(${BENCHMARK_NAME} "${BENCHMARK_SRC}")
  target_include_directories(
    ${BENCHMARK_NAME} PRIVATE "$<BUILD_INTERFACE:${Morpheus_SOURCE_DIR}>")
  target_link_libraries(${BENCHMARK_NAME} morpheus)
  set_target_properties(
    ${BENCHMARK_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               RUNTIME_OUTPUT_DIRECTORY
               "$<BUILD_INTERFACE:${Morpheus_BINARY_DIR}/benchmarks>")
endfunction()

# These examples use the separate compilation
set(SOURCES_IDIOMATIC_BENCHMARKS # TODO: Fill in with source files
)

string(REPLACE ".cpp" "" BASENAMES_IDIOMATIC_BENCHMARKS
               "${SOURCES_IDIOMATIC_BENCHMARKS}")
set(TARGETS_IDIOMATIC_BENCHMARKS ${BASENAMES_IDIOMATIC_BENCHMARKS})

foreach(name ${TARGETS_IDIOMATIC_BENCHMARKS})
  configurebenchmark(${name} ${Morpheus_BENCHMARKS_DIR}/${name}.cpp)
endforeach()
