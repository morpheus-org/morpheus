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
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/category_files)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/format_files)

foreach(Tag Serial;OpenMP;Cuda;HIP)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)

  set(${Tag}_SOURCES)
  foreach(Fmt Coo;Csr)
    string(TOUPPER ${Fmt} FORMAT)
    string(TOLOWER ${Fmt} fmtdir)
    if(Morpheus_ENABLE_${DEVICE})
      set(testdir ${CMAKE_CURRENT_BINARY_DIR}/${dir}/${fmtdir})
      file(MAKE_DIRECTORY ${testdir})

      foreach(Name Multiply Copy)
        set(file ${testdir}/Test${Tag}_${Fmt}_${Name}.cpp)
        # Write to a temporary intermediate file and call configure_file to
        # avoid updating timestamps triggering unnecessary rebuilds on
        # subsequent cmake runs.
        file(
          WRITE ${testdir}/dummy.cpp
          "#include <Test${Tag}_Category.hpp>\n"
          "#include <Test${Fmt}Matrix_Format.hpp>\n"
          "#define TESTSUITE_NAME ${Tag}${Fmt}\n"
          "#include <Test_${Name}.hpp>\n")
        configure_file(${testdir}/dummy.cpp ${file})
        list(APPEND ${Tag}_SOURCES ${file})
      endforeach()
    endif()
  endforeach()
endforeach()

if(Morpheus_ENABLE_SERIAL)
  morpheus_add_executable_and_test(UnitTest_Serial SOURCES TestMain.cpp
                                   ${Serial_SOURCES})
endif()

if(Morpheus_ENABLE_OPENMP)
  morpheus_add_executable_and_test(UnitTest_OpenMP SOURCES TestMain.cpp
                                   ${OpenMP_SOURCES})
endif()

if(Morpheus_ENABLE_CUDA)
  morpheus_add_executable_and_test(UnitTest_Cuda SOURCES TestMain.cpp
                                   ${Cuda_SOURCES})
endif()

if(Morpheus_ENABLE_HIP)
  morpheus_add_executable_and_test(UnitTest_HIP SOURCES TestMain.cpp
                                   ${HIP_SOURCES})
endif()
