MORPHEUS_CFG_DEPENDS(CXX_STD COMPILER_ID)

FUNCTION(morpheus_set_cxx_standard_feature standard)
  SET(EXTENSION_NAME CMAKE_CXX${standard}_EXTENSION_COMPILE_OPTION)
  SET(STANDARD_NAME  CMAKE_CXX${standard}_STANDARD_COMPILE_OPTION)
  SET(FEATURE_NAME   cxx_std_${standard})
  #CMake's way of telling us that the standard (or extension)
  #flags are supported is the extension/standard variables
  IF (NOT DEFINED CMAKE_CXX_EXTENSIONS)
    IF(MORPHEUS_DONT_ALLOW_EXTENSIONS)
      GLOBAL_SET(MORPHEUS_USE_CXX_EXTENSIONS OFF)
    ELSE()
      GLOBAL_SET(MORPHEUS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    IF(MORPHEUS_DONT_ALLOW_EXTENSIONS)
      MESSAGE(FATAL_ERROR "The chosen configuration does not support CXX extensions flags: ${MORPHEUS_DONT_ALLOW_EXTENSIONS}. Must set CMAKE_CXX_EXTENSIONS=OFF to continue")
    ELSE()
      GLOBAL_SET(MORPHEUS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSE()
    #For trilinos, we need to make sure downstream projects
    GLOBAL_SET(MORPHEUS_USE_CXX_EXTENSIONS OFF)
  ENDIF()

  IF (MORPHEUS_USE_CXX_EXTENSIONS AND ${EXTENSION_NAME})
    MESSAGE(STATUS "Using ${${EXTENSION_NAME}} for C++${standard} extensions as feature")
    GLOBAL_SET(MORPHEUS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSEIF(NOT MORPHEUS_USE_CXX_EXTENSIONS AND ${STANDARD_NAME})
    MESSAGE(STATUS "Using ${${STANDARD_NAME}} for C++${standard} standard as feature")
    # IF (MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA AND (MORPHEUS_CXX_HOST_COMPILER_ID STREQUAL GNU OR MORPHEUS_CXX_HOST_COMPILER_ID STREQUAL Clang))
    #   SET(SUPPORTED_NVCC_FLAGS "-std=c++17")
    #   IF (NOT ${${STANDARD_NAME}} IN_LIST SUPPORTED_NVCC_FLAGS)
    #     MESSAGE(FATAL_ERROR "CMake wants to use ${${STANDARD_NAME}} which is not supported by NVCC. Using a more recent host compiler or a more recent CMake version might help.")
    #   ENDIF()
    # ENDIF()
    GLOBAL_SET(MORPHEUS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
    #MSVC doesn't need a command line flag, that doesn't mean it has no support
    MESSAGE(STATUS "Using no flag for C++${standard} standard as feature")
    GLOBAL_SET(MORPHEUS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
#   ELSEIF((MORPHEUS_CXX_COMPILER_ID STREQUAL "NVIDIA") AND WIN32)
#     MESSAGE(STATUS "Using no flag for C++${standard} standard as feature")
#     GLOBAL_SET(MORPHEUS_CXX_STANDARD_FEATURE "")
  ELSE()
    #nope, we can't do anything here
    MESSAGE(WARNING "C++${standard} is not supported as a compiler feature. We will choose custom flags for now, but this behavior has been deprecated.")
    GLOBAL_SET(MORPHEUS_CXX_STANDARD_FEATURE "")
  ENDIF()
ENDFUNCTION()


IF (MORPHEUS_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  #make sure these are consistent
  IF (NOT MORPHEUS_CXX_STANDARD STREQUAL CMAKE_CXX_STANDARD)
    MESSAGE(WARNING "Specified both CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} and MORPHEUS_CXX_STANDARD=${MORPHEUS_CXX_STANDARD}, but they don't match")
    SET(CMAKE_CXX_STANDARD ${MORPHEUS_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
  ENDIF()
ENDIF()


IF(MORPHEUS_CXX_STANDARD STREQUAL "17")
  morpheus_set_cxx_standard_feature(17)
  SET(MORPHEUS_CXX_INTERMEDIATE_STANDARD "1Z")
  SET(MORPHEUS_ENABLE_CXX17 ON)
ELSEIF(MORPHEUS_CXX_STANDARD STREQUAL "20")
  morpheus_set_cxx_standard_feature(20)
  SET(MORPHEUS_CXX_INTERMEDIATE_STANDARD "2A")
  SET(MORPHEUS_ENABLE_CXX20 ON)
ELSEIF(MORPHEUS_CXX_STANDARD STREQUAL "98" OR MORPHEUS_CXX_STANDARD STREQUAL "11" OR MORPHEUS_CXX_STANDARD STREQUAL "14")
  MESSAGE(FATAL_ERROR "Morpheus requires C++17 or newer!")
ELSE()
  MESSAGE(FATAL_ERROR "Unknown C++ standard ${MORPHEUS_CXX_STANDARD} - must be 17, or 20")
ENDIF()

# Enforce that we can compile a simple C++17 program

TRY_COMPILE(CAN_COMPILE_CPP17
  ${MORPHEUS_TOP_BUILD_DIR}/corner_cases
  ${MORPHEUS_SOURCE_DIR}/cmake/compile_tests/cplusplus17.cpp
  OUTPUT_VARIABLE ERROR_MESSAGE
  CXX_STANDARD 17
)
if (NOT CAN_COMPILE_CPP17)
  UNSET(CAN_COMPILE_CPP17 CACHE) #make sure CMake always re-runs this
  MESSAGE(FATAL_ERROR "C++${MORPHEUS_CXX_STANDARD}-compliant compiler detected, but unable to compile C++17 or later program. Verify that ${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION} is set up correctly (e.g., check that correct library headers are being used).\nFailing output:\n ${ERROR_MESSAGE}")
ENDIF()
UNSET(CAN_COMPILE_CPP17 CACHE) #make sure CMake always re-runs this


# Enforce that extensions are turned off for nvcc_wrapper.
# For compiling CUDA code using nvcc_wrapper, we will use the host compiler's
# flags for turning on C++17.  Since for compiler ID and versioning purposes
# CMake recognizes the host compiler when calling nvcc_wrapper, this just
# works.  Both NVCC and nvcc_wrapper only recognize '-std=c++14' which means
# that we can only use host compilers for CUDA builds that use those flags.
# It also means that extensions (gnu++14) can't be turned on for CUDA builds.

IF(MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    SET(CMAKE_CXX_EXTENSIONS OFF)
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    MESSAGE(FATAL_ERROR "NVCC doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
  ENDIF()
ENDIF()

IF(MORPHEUS_ENABLE_CUDA)
  # ENFORCE that the compiler can compile CUDA code.
  IF(MORPHEUS_CXX_COMPILER_ID STREQUAL Clang)
    IF(MORPHEUS_CXX_COMPILER_VERSION VERSION_LESS 4.0.0)
      MESSAGE(FATAL_ERROR "Compiling CUDA code directly with Clang requires version 4.0.0 or higher.")
    ENDIF()
    IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
      SET(CMAKE_CXX_EXTENSIONS OFF)
    ELSEIF(CMAKE_CXX_EXTENSIONS)
      MESSAGE(FATAL_ERROR "Compiling CUDA code with clang doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
    ENDIF()
  ELSEIF(NOT MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
    MESSAGE(FATAL_ERROR "Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang or use kokkos_launch_compiler, but compiler ID was ${MORPHEUS_CXX_COMPILER_ID}")
  ENDIF()
ENDIF()

IF (NOT MORPHEUS_CXX_STANDARD_FEATURE)
  #we need to pick the C++ flags ourselves
  UNSET(CMAKE_CXX_STANDARD)
  UNSET(CMAKE_CXX_STANDARD CACHE)
  IF(MORPHEUS_CXX_COMPILER_ID STREQUAL Cray)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/cray.cmake)
    morpheus_set_cray_flags(${MORPHEUS_CXX_STANDARD} ${MORPHEUS_CXX_INTERMEDIATE_STANDARD})
#   ELSEIF(MORPHEUS_CXX_COMPILER_ID STREQUAL PGI)
#     INCLUDE(${MORPHEUS_SRC_PATH}/cmake/pgi.cmake)
#     morpheus_set_pgi_flags(${MORPHEUS_CXX_STANDARD} ${MORPHEUS_CXX_INTERMEDIATE_STANDARD})
  ELSEIF(MORPHEUS_CXX_COMPILER_ID STREQUAL Intel)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/intel.cmake)
    morpheus_set_intel_flags(${MORPHEUS_CXX_STANDARD} ${MORPHEUS_CXX_INTERMEDIATE_STANDARD})
#   ELSEIF((MORPHEUS_CXX_COMPILER_ID STREQUAL "MSVC") OR ((MORPHEUS_CXX_COMPILER_ID STREQUAL "NVIDIA") AND WIN32))
#     INCLUDE(${MORPHEUS_SRC_PATH}/cmake/msvc.cmake)
#     morpheus_set_msvc_flags(${MORPHEUS_CXX_STANDARD} ${MORPHEUS_CXX_INTERMEDIATE_STANDARD})
  ELSE()
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/gnu.cmake)
    morpheus_set_gnu_flags(${MORPHEUS_CXX_STANDARD} ${MORPHEUS_CXX_INTERMEDIATE_STANDARD})
  ENDIF()
  #check that the compiler accepts the C++ standard flag
  INCLUDE(CheckCXXCompilerFlag)
  IF (DEFINED CXX_STD_FLAGS_ACCEPTED)
    UNSET(CXX_STD_FLAGS_ACCEPTED CACHE)
  ENDIF()
  CHECK_CXX_COMPILER_FLAG("${MORPHEUS_CXX_STANDARD_FLAG}" CXX_STD_FLAGS_ACCEPTED)
  IF (NOT CXX_STD_FLAGS_ACCEPTED)
    CHECK_CXX_COMPILER_FLAG("${MORPHEUS_CXX_INTERMEDIATE_STANDARD_FLAG}" CXX_INT_STD_FLAGS_ACCEPTED)
    IF (NOT CXX_INT_STD_FLAGS_ACCEPTED)
      MESSAGE(FATAL_ERROR "${MORPHEUS_CXX_COMPILER_ID} did not accept ${MORPHEUS_CXX_STANDARD_FLAG} or ${MORPHEUS_CXX_INTERMEDIATE_STANDARD_FLAG}. You likely need to reduce the level of the C++ standard from ${MORPHEUS_CXX_STANDARD}")
    ENDIF()
    SET(MORPHEUS_CXX_STANDARD_FLAG ${MORPHEUS_CXX_INTERMEDIATE_STANDARD_FLAG})
  ENDIF()
  MESSAGE(STATUS "Compiler features not supported, but ${MORPHEUS_CXX_COMPILER_ID} accepts ${MORPHEUS_CXX_STANDARD_FLAG}")
ENDIF()