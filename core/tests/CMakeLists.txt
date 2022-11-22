message(STATUS "Core Tests are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/category_files)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/setup)

foreach(Tag Serial;OpenMP;Cuda;HIP)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)
  global_set(${Tag}_SOURCES)

  if(MORPHEUS_ENABLE_${DEVICE})
    set(testdir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
    file(MAKE_DIRECTORY ${testdir})

    foreach(
      Name
      MatrixBase
      DenseVector
      DenseVector_Binary
      DenseVector_CompatibleBinary
      DenseMatrix
      DenseMatrix_Binary
      DenseMatrix_CompatibleBinary
      CooMatrix
      CooMatrix_Binary
      CooMatrix_CompatibleBinary
      CooMatrix_CompatibleDynamicBinary
      CsrMatrix
      CsrMatrix_Binary
      CsrMatrix_CompatibleBinary
      CsrMatrix_CompatibleDynamicBinary
      DiaMatrix
      DiaMatrix_Binary
      DiaMatrix_CompatibleBinary
      DiaMatrix_CompatibleDynamicBinary
      DynamicMatrix
      Dot
      TypeTraits
      Metaprogramming
      ContainerFactory
      FormatTags
      GenericSpace)
      set(file ${testdir}/Test_${Tag}_${Name}.cpp)
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake
      # runs.
      file(WRITE ${testdir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n"
                                      "#include <Test_${Name}.hpp>\n")
      configure_file(${testdir}/dummy.cpp ${file})
      global_append(${Tag}_SOURCES ${file})
    endforeach()
  endif()
endforeach()

set(ALL_SOURCES)

if(Morpheus_ENABLE_SERIAL)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_Serial SOURCES TestMain.cpp
                                     ${Serial_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${Serial_SOURCES})
endif()

if(Morpheus_ENABLE_OPENMP)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_OpenMP SOURCES TestMain.cpp
                                     ${OpenMP_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${OpenMP_SOURCES})
endif()

if(Morpheus_ENABLE_CUDA)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_Cuda SOURCES TestMain.cpp
                                     ${Cuda_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${Cuda_SOURCES})
endif()

if(Morpheus_ENABLE_HIP)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_HIP SOURCES TestMain.cpp
                                     ${HIP_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${HIP_SOURCES})
endif()

if(NOT Morpheus_ENABLE_INDIVIDUAL_TESTS)
  morpheus_add_executable_and_test(UnitTest SOURCES TestMain.cpp ${ALL_SOURCES})
endif()
