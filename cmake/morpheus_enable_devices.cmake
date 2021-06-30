FUNCTION(MORPHEUS_DEVICE_OPTION SUFFIX DEFAULT DEV_TYPE DOCSTRING)
  MORPHEUS_OPTION(ENABLE_${SUFFIX} ${DEFAULT} BOOL ${DOCSTRING})
  STRING(TOUPPER ${SUFFIX} UC_NAME)
  IF (MORPHEUS_ENABLE_${UC_NAME})
    LIST(APPEND MORPHEUS_ENABLED_DEVICES    ${SUFFIX})
    #I hate that CMake makes me do this
    SET(MORPHEUS_ENABLED_DEVICES    ${MORPHEUS_ENABLED_DEVICES}    PARENT_SCOPE)
  ENDIF()
  SET(MORPHEUS_ENABLE_${UC_NAME} ${MORPHEUS_ENABLE_${UC_NAME}} PARENT_SCOPE)
  IF (MORPHEUS_ENABLE_${UC_NAME} AND DEV_TYPE STREQUAL "HOST")
    SET(MORPHEUS_HAS_HOST ON PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

MORPHEUS_CFG_DEPENDS(DEVICES NONE)

# Put a check in just in case people are using this option
MORPHEUS_DEPRECATED_LIST(DEVICES ENABLE)

# detect clang++ / cl / clang-cl clashes
IF (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
  # this specific test requires CMake >= 3.15
  IF ("x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xGNU")
    # use pure clang++ instead of clang-cl
    SET(MORPHEUS_COMPILER_CLANG_MSVC OFF)
  ELSE()
    # it defaults to clang-cl
    SET(MORPHEUS_COMPILER_CLANG_MSVC ON)
  ENDIF()
ENDIF()


SET(OMP_DEFAULT OFF)
MORPHEUS_DEVICE_OPTION(OPENMP ${OMP_DEFAULT} HOST "Whether to build OpenMP backend")
IF(MORPHEUS_ENABLE_OPENMP)
  SET(ClangOpenMPFlag -fopenmp=libomp)
  IF(MORPHEUS_CLANG_IS_CRAY)
    SET(ClangOpenMPFlag -fopenmp)
  ENDIF()
  IF(MORPHEUS_COMPILER_CLANG_MSVC)
    #for clang-cl expression /openmp yields an error, so directly add the specific Clang flag
    SET(ClangOpenMPFlag /clang:-fopenmp=libomp)
  ENDIF()
  IF(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    #link omp library from LLVM lib dir, no matter if it is clang-cl or clang++
    get_filename_component(LLVM_BIN_DIR ${CMAKE_CXX_COMPILER_AR} DIRECTORY)
    COMPILER_SPECIFIC_LIBS(Clang "${LLVM_BIN_DIR}/../lib/libomp.lib")
  ENDIF()
  IF(MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
    COMPILER_SPECIFIC_FLAGS(
      COMPILER_ID MORPHEUS_CXX_HOST_COMPILER_ID
      Clang      -Xcompiler ${ClangOpenMPFlag}
      IntelLLVM  -Xcompiler -fiopenmp
      PGI        -Xcompiler -mp
      Cray       NO-VALUE-SPECIFIED
      XL         -Xcompiler -qsmp=omp
      DEFAULT    -Xcompiler -fopenmp
    )
  ELSE()
    COMPILER_SPECIFIC_FLAGS(
      Clang      ${ClangOpenMPFlag}
      IntelLLVM  -fiopenmp
      AppleClang -Xpreprocessor -fopenmp
      PGI        -mp
      Cray       NO-VALUE-SPECIFIED
      XL         -qsmp=omp
      DEFAULT    -fopenmp
    )
    COMPILER_SPECIFIC_LIBS(
      AppleClang -lomp
    )
  ENDIF()
ENDIF()



# SET(CUDA_DEFAULT OFF)
# MORPHEUS_DEVICE_OPTION(CUDA ${CUDA_DEFAULT} DEVICE "Whether to build CUDA backend")

# IF (MORPHEUS_ENABLE_CUDA)
#   GLOBAL_SET(MORPHEUS_DONT_ALLOW_EXTENSIONS "CUDA enabled")
#   IF(WIN32 AND NOT MORPHEUS_CXX_COMPILER_ID STREQUAL Clang)
#     GLOBAL_APPEND(MORPHEUS_COMPILE_OPTIONS -x cu)
#   ENDIF()
# ## Cuda has extra setup requirements, turn on Morpheus_Setup_Cuda.hpp in macros
#   LIST(APPEND DEVICE_SETUP_LIST Cuda)
# ENDIF()

# We want this to default to OFF for cache reasons, but if no
# host space is given, then activate serial
IF (MORPHEUS_HAS_HOST)
  SET(SERIAL_DEFAULT OFF)
ELSE()
  SET(SERIAL_DEFAULT ON)
  IF (NOT DEFINED Morpheus_ENABLE_SERIAL)
    MESSAGE(STATUS "SERIAL backend is being turned on to ensure there is at least one Host space. To change this, you must enable another host execution space and configure with -DMorpheus_ENABLE_SERIAL=OFF or change CMakeCache.txt")
  ENDIF()
ENDIF()
MORPHEUS_DEVICE_OPTION(SERIAL ${SERIAL_DEFAULT} HOST "Whether to build serial backend")

# MORPHEUS_DEVICE_OPTION(HIP OFF DEVICE "Whether to build HIP backend")

# ## HIP has extra setup requirements, turn on Morpheus_Setup_HIP.hpp in macros
# IF (MORPHEUS_ENABLE_HIP)
#   LIST(APPEND DEVICE_SETUP_LIST HIP)
# ENDIF()