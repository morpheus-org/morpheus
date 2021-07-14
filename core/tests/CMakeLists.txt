message(STATUS "Core Tests are enabled")

#
# Add test-only library for gtest to be reused by all the subpackages
#

set(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/gtest)

# need here for tribits
morpheus_include_directories(${GTEST_SOURCE_DIR})
morpheus_add_test_library(
  morpheus_gtest HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h SOURCES
  ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)

# avoid deprecation warnings from MSVC
target_compile_definitions(morpheus_gtest PUBLIC GTEST_HAS_TR1_TUPLE=0
                                                 GTEST_HAS_PTHREAD=0)

target_include_directories(morpheus_gtest PUBLIC ${GTEST_SOURCE_DIR})
if((NOT (Morpheus_ENABLE_CUDA AND WIN32)) AND (NOT ("${KOKKOS_CXX_COMPILER_ID}"
                                                    STREQUAL "Fujitsu")))
  target_compile_features(morpheus_gtest PUBLIC cxx_std_14)
endif()

# Suppress clang-tidy diagnostics on code that we do not have control over
if(CMAKE_CXX_CLANG_TIDY)
  set_target_properties(morpheus_gtest PROPERTIES CXX_CLANG_TIDY "")
endif()

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
