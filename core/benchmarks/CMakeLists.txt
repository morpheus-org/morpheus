message(STATUS "Core Benchmarks are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

morpheus_add_executable_and_test(Benchmarks_Convert SOURCES
                                 Benchmarks_Convert.cpp)

morpheus_add_executable_and_test(Benchmarks_Copy SOURCES Benchmarks_Copy.cpp)

morpheus_add_executable_and_test(Benchmarks_Convert SOURCES
                                 Benchmarks_Convert.cpp)
