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
if((NOT (MORPHEUS_ENABLE_CUDA AND WIN32)) AND (NOT ("${KOKKOS_CXX_COMPILER_ID}"
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
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/Cuda)

# Unary tests
foreach(Fmt DenseVector;DenseMatrix;Coo;Csr;Dia)
  string(TOUPPER ${Fmt} FORMAT)
  string(TOLOWER ${Fmt} fmtdir)

  foreach(Tag Serial;OpenMP;Cuda;HIP)
    string(TOUPPER ${Tag} DEVICE)
    string(TOLOWER ${Tag} dir)

    set(${Fmt}_${Tag}_UNARY_SOURCES)
    if(MORPHEUS_ENABLE_${DEVICE})
      set(testdir ${CMAKE_CURRENT_BINARY_DIR}/unary/${fmtdir}/${dir})
      file(MAKE_DIRECTORY ${testdir})

      foreach(Name ContainerTraits)
        set(file ${testdir}/Test${Tag}_${Fmt}_${Name}.cpp)
        # Write to a temporary intermediate file and call configure_file to
        # avoid updating timestamps triggering unnecessary rebuilds on
        # subsequent cmake runs.
        file(
          WRITE ${testdir}/dummy.cpp
          "#include <Test${Tag}_Category.hpp>\n"
          "#include <Test${Fmt}_Format.hpp>\n"
          "#define TESTSUITE_NAME ${Tag}${Fmt}\n"
          "#include <Test_${Name}.hpp>\n")
        configure_file(${testdir}/dummy.cpp ${file})
        list(APPEND ${Fmt}_${Tag}_UNARY_SOURCES ${file})
      endforeach()
    endif()
  endforeach()
endforeach()

# Do test independent of formats
foreach(Fmt DenseVector)
  string(TOUPPER ${Fmt} FORMAT)
  string(TOLOWER ${Fmt} fmtdir)

  foreach(Tag Serial;OpenMP;Cuda;HIP)
    string(TOUPPER ${Tag} DEVICE)
    string(TOLOWER ${Tag} dir)

    set(${Fmt}_${Tag}_INDEPENDENT_SOURCES)
    if(MORPHEUS_ENABLE_${DEVICE})
      set(testdir ${CMAKE_CURRENT_BINARY_DIR}/independent/${dir})
      file(MAKE_DIRECTORY ${testdir})

      foreach(Name MirrorContainers)
        set(file ${testdir}/Test${Tag}_${Fmt}_${Name}.cpp)
        # Write to a temporary intermediate file and call configure_file to
        # avoid updating timestamps triggering unnecessary rebuilds on
        # subsequent cmake runs.
        file(
          WRITE ${testdir}/dummy.cpp
          "#include <Test${Tag}_Category.hpp>\n"
          "#include <Test${Fmt}_Format.hpp>\n"
          "#define TESTSUITE_NAME ${Tag}${Fmt}\n"
          "#include <Test_${Name}.hpp>\n")
        configure_file(${testdir}/dummy.cpp ${file})
        list(APPEND ${Fmt}_${Tag}_INDEPENDENT_SOURCES ${file})
      endforeach()
    endif()
  endforeach()
endforeach()

# TODO: Create a single build for all formats per device
foreach(Fmt DenseVector;DenseMatrix;Coo;Csr;Dia)
  if(MORPHEUS_ENABLE_SERIAL)
    morpheus_add_executable_and_test(
      UnitTest_${Fmt}_Serial SOURCES TestMain.cpp
      ${${Fmt}_Serial_UNARY_SOURCES} ${${Fmt}_Serial_INDEPENDENT_SOURCES})
  endif()

  if(MORPHEUS_ENABLE_OPENMP)
    morpheus_add_executable_and_test(
      UnitTest_${Fmt}_OpenMP SOURCES TestMain.cpp
      ${${Fmt}_OpenMP_UNARY_SOURCES} ${${Fmt}_OpenMP_INDEPENDENT_SOURCES})
  endif()

  if(MORPHEUS_ENABLE_CUDA)
    morpheus_add_executable_and_test(
      UnitTest_${Fmt}_Cuda SOURCES TestMain.cpp ${${Fmt}_Cuda_UNARY_SOURCES}
      ${${Fmt}_Cuda_INDEPENDENT_SOURCES})
  endif()

  if(MORPHEUS_ENABLE_HIP)
    morpheus_add_executable_and_test(
      UnitTest_${Fmt}_HIP SOURCES TestMain.cpp ${${Fmt}_HIP_UNARY_SOURCES}
      ${${Fmt}_HIP_INDEPENDENT_SOURCES})
  endif()
endforeach()
