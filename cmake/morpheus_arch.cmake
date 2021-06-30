
FUNCTION(MORPHEUS_ARCH_OPTION SUFFIX DEV_TYPE DESCRIPTION)
#all optimizations off by default
MORPHEUS_OPTION(ARCH_${SUFFIX} OFF BOOL "Optimize for ${DESCRIPTION} (${DEV_TYPE})")
SET(MORPHEUS_ARCH_${SUFFIX} ${MORPHEUS_ARCH_${SUFFIX}} PARENT_SCOPE)
SET(MORPHEUS_OPTION_KEYS ${MORPHEUS_OPTION_KEYS} PARENT_SCOPE)
SET(MORPHEUS_OPTION_VALUES ${MORPHEUS_OPTION_VALUES} PARENT_SCOPE)
SET(MORPHEUS_OPTION_TYPES ${MORPHEUS_OPTION_TYPES} PARENT_SCOPE)
IF(MORPHEUS_ARCH_${SUFFIX})
  LIST(APPEND MORPHEUS_ENABLED_ARCH_LIST ${SUFFIX})
  SET(MORPHEUS_ENABLED_ARCH_LIST ${MORPHEUS_ENABLED_ARCH_LIST} PARENT_SCOPE)
ENDIF()
ENDFUNCTION()


# Make sure devices and compiler ID are done
MORPHEUS_CFG_DEPENDS(ARCH COMPILER_ID)
MORPHEUS_CFG_DEPENDS(ARCH DEVICES)
MORPHEUS_CFG_DEPENDS(ARCH OPTIONS)

MORPHEUS_CHECK_DEPRECATED_OPTIONS(
ARCH_EPYC   "Please replace EPYC with ZEN or ZEN2, depending on your platform"
ARCH_RYZEN  "Please replace RYZEN with ZEN or ZEN2, depending on your platform"
)

#-------------------------------------------------------------------------------
# List of possible host architectures.
#-------------------------------------------------------------------------------
SET(MORPHEUS_ARCH_LIST)


MORPHEUS_DEPRECATED_LIST(ARCH ARCH)
MORPHEUS_ARCH_OPTION(AMDAVX          HOST "AMD chip")
MORPHEUS_ARCH_OPTION(ARMV80          HOST "ARMv8.0 Compatible CPU")
MORPHEUS_ARCH_OPTION(ARMV81          HOST "ARMv8.1 Compatible CPU")
MORPHEUS_ARCH_OPTION(ARMV8_THUNDERX  HOST "ARMv8 Cavium ThunderX CPU")
MORPHEUS_ARCH_OPTION(ARMV8_THUNDERX2 HOST "ARMv8 Cavium ThunderX2 CPU")
MORPHEUS_ARCH_OPTION(A64FX           HOST "ARMv8.2 with SVE Support")
MORPHEUS_ARCH_OPTION(WSM             HOST "Intel Westmere CPU")
MORPHEUS_ARCH_OPTION(SNB             HOST "Intel Sandy/Ivy Bridge CPUs")
MORPHEUS_ARCH_OPTION(HSW             HOST "Intel Haswell CPUs")
MORPHEUS_ARCH_OPTION(BDW             HOST "Intel Broadwell Xeon E-class CPUs")
MORPHEUS_ARCH_OPTION(SKX             HOST "Intel Sky Lake Xeon E-class HPC CPUs (AVX512)")
MORPHEUS_ARCH_OPTION(PASCAL60        GPU  "NVIDIA Pascal generation CC 6.0")
MORPHEUS_ARCH_OPTION(PASCAL61        GPU  "NVIDIA Pascal generation CC 6.1")
MORPHEUS_ARCH_OPTION(VOLTA70         GPU  "NVIDIA Volta generation CC 7.0")
MORPHEUS_ARCH_OPTION(VOLTA72         GPU  "NVIDIA Volta generation CC 7.2")
MORPHEUS_ARCH_OPTION(TURING75        GPU  "NVIDIA Turing generation CC 7.5")
MORPHEUS_ARCH_OPTION(AMPERE80        GPU  "NVIDIA Ampere generation CC 8.0")
MORPHEUS_ARCH_OPTION(AMPERE86        GPU  "NVIDIA Ampere generation CC 8.6")
MORPHEUS_ARCH_OPTION(ZEN             HOST "AMD Zen architecture")
MORPHEUS_ARCH_OPTION(ZEN2            HOST "AMD Zen2 architecture")
MORPHEUS_ARCH_OPTION(ZEN3            HOST "AMD Zen3 architecture")
MORPHEUS_ARCH_OPTION(VEGA900         GPU  "AMD GPU MI25 GFX900")
MORPHEUS_ARCH_OPTION(VEGA906         GPU  "AMD GPU MI50/MI60 GFX906")
MORPHEUS_ARCH_OPTION(VEGA908         GPU  "AMD GPU MI100 GFX908")
MORPHEUS_ARCH_OPTION(INTEL_GEN       GPU  "Intel GPUs Gen9+")

IF(MORPHEUS_ENABLE_COMPILER_WARNINGS)
SET(COMMON_WARNINGS
  "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
  "-Wsign-compare" "-Wtype-limits" "-Wuninitialized")

SET(GNU_WARNINGS "-Wempty-body" "-Wclobbered" "-Wignored-qualifiers"
  ${COMMON_WARNINGS})

COMPILER_SPECIFIC_FLAGS(
  COMPILER_ID CMAKE_CXX_COMPILER_ID
#   PGI         NO-VALUE-SPECIFIED
  GNU         ${GNU_WARNINGS}
  DEFAULT     ${COMMON_WARNINGS}
)
ENDIF()


# #------------------------------- MORPHEUS_CUDA_OPTIONS ---------------------------
# #clear anything that might be in the cache
# GLOBAL_SET(MORPHEUS_CUDA_OPTIONS)
# # Construct the Makefile options
# IF (MORPHEUS_CXX_COMPILER_ID STREQUAL Clang)
#     SET(CUDA_ARCH_FLAG "--cuda-gpu-arch")
#     GLOBAL_APPEND(MORPHEUS_CUDA_OPTIONS -x cuda)
#     # MORPHEUS_CUDA_DIR has priority over CUDAToolkit_BIN_DIR
#     IF (MORPHEUS_CUDA_DIR)
#     GLOBAL_APPEND(MORPHEUS_CUDA_OPTIONS --cuda-path=${Morpheus_CUDA_DIR})
#     ELSEIF(CUDAToolkit_BIN_DIR)
#     GLOBAL_APPEND(MORPHEUS_CUDA_OPTIONS --cuda-path=${CUDAToolkit_BIN_DIR}/..)
#     ENDIF()
# ELSEIF(MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
#     SET(CUDA_ARCH_FLAG "-arch")
# ENDIF()

# IF (MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
#     STRING(TOUPPER "${CMAKE_BUILD_TYPE}" _UPPERCASE_CMAKE_BUILD_TYPE)
#     IF (MORPHEUS_ENABLE_DEBUG OR _UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#         GLOBAL_APPEND(MORPHEUS_CUDA_OPTIONS -lineinfo)
#     ENDIF()

#     UNSET(_UPPERCASE_CMAKE_BUILD_TYPE)
#     IF (MORPHEUS_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0 AND MORPHEUS_CXX_COMPILER_VERSION VERSION_LESS 10.0)
#         GLOBAL_APPEND(MORPHEUS_CUDAFE_OPTIONS --diag_suppress=esa_on_defaulted_function_ignored)
#     ENDIF()
# ENDIF()


#------------------------------- MORPHEUS_HIP_OPTIONS ---------------------------
#clear anything that might be in the cache
GLOBAL_SET(MORPHEUS_AMDGPU_OPTIONS)
IF(MORPHEUS_ENABLE_HIP)
    IF(MORPHEUS_CXX_COMPILER_ID STREQUAL HIPCC)
        SET(AMDGPU_ARCH_FLAG "--amdgpu-target")
    ELSE()
        SET(AMDGPU_ARCH_FLAG "--offload-arch")
        GLOBAL_APPEND(MORPHEUS_AMDGPU_OPTIONS -x hip)
        IF(DEFINED ENV{ROCM_PATH})
            GLOBAL_APPEND(MORPHEUS_AMDGPU_OPTIONS --rocm-path=$ENV{ROCM_PATH})
        ENDIF()
    ENDIF()
ENDIF()


IF (MORPHEUS_ARCH_ARMV80)
COMPILER_SPECIFIC_FLAGS(
  Cray NO-VALUE-SPECIFIED
#   PGI  NO-VALUE-SPECIFIED
  DEFAULT -march=armv8-a
)
ENDIF()

IF (MORPHEUS_ARCH_ARMV81)
COMPILER_SPECIFIC_FLAGS(
  Cray NO-VALUE-SPECIFIED
#   PGI  NO-VALUE-SPECIFIED
  DEFAULT -march=armv8.1-a
)
ENDIF()

IF (MORPHEUS_ARCH_ARMV8_THUNDERX)
SET(MORPHEUS_ARCH_ARMV80 ON) #Not a cache variable
COMPILER_SPECIFIC_FLAGS(
  Cray NO-VALUE-SPECIFIED
#   PGI  NO-VALUE-SPECIFIED
  DEFAULT -march=armv8-a -mtune=thunderx
)
ENDIF()

IF (MORPHEUS_ARCH_ARMV8_THUNDERX2)
SET(MORPHEUS_ARCH_ARMV81 ON) #Not a cache variable
COMPILER_SPECIFIC_FLAGS(
  Cray NO-VALUE-SPECIFIED
#   PGI  NO-VALUE-SPECIFIED
  DEFAULT -mcpu=thunderx2t99 -mtune=thunderx2t99
)
ENDIF()

IF (MORPHEUS_ARCH_A64FX)
COMPILER_SPECIFIC_FLAGS(
  DEFAULT -march=armv8.2-a+sve
  Clang -march=armv8.2-a+sve -msve-vector-bits=512
  GCC -march=armv8.2-a+sve -msve-vector-bits=512
)
ENDIF()

IF (MORPHEUS_ARCH_ZEN)
COMPILER_SPECIFIC_FLAGS(
  Intel   -mavx2
  DEFAULT -march=znver1 -mtune=znver1
)
SET(MORPHEUS_ARCH_AMD_ZEN  ON)
SET(MORPHEUS_ARCH_AMD_AVX2 ON)
ENDIF()

IF (MORPHEUS_ARCH_ZEN2)
COMPILER_SPECIFIC_FLAGS(
  Intel   -mavx2
  DEFAULT -march=znver2 -mtune=znver2
)
SET(MORPHEUS_ARCH_AMD_ZEN2 ON)
SET(MORPHEUS_ARCH_AMD_AVX2 ON)
ENDIF()

IF (MORPHEUS_ARCH_ZEN3)
COMPILER_SPECIFIC_FLAGS(
  Intel   -mavx2
  DEFAULT -march=znver3 -mtune=znver3
)
SET(MORPHEUS_ARCH_AMD_ZEN3 ON)
SET(MORPHEUS_ARCH_AMD_AVX2 ON)
ENDIF()

IF (MORPHEUS_ARCH_WSM)
COMPILER_SPECIFIC_FLAGS(
  Intel   -xSSE4.2
#   PGI     -tp=nehalem
  Cray    NO-VALUE-SPECIFIED
  DEFAULT -msse4.2
)
SET(MORPHEUS_ARCH_SSE42 ON)
ENDIF()

IF (MORPHEUS_ARCH_SNB OR MORPHEUS_ARCH_AMDAVX)
SET(MORPHEUS_ARCH_AVX ON)
COMPILER_SPECIFIC_FLAGS(
  Intel   -mavx
#   PGI     -tp=sandybridge
  Cray    NO-VALUE-SPECIFIED
  DEFAULT -mavx
)
ENDIF()

IF (MORPHEUS_ARCH_HSW)
SET(MORPHEUS_ARCH_AVX2 ON)
COMPILER_SPECIFIC_FLAGS(
  Intel   -xCORE-AVX2
#   PGI     -tp=haswell
  Cray    NO-VALUE-SPECIFIED
  DEFAULT -march=core-avx2 -mtune=core-avx2
)
ENDIF()

IF (MORPHEUS_ARCH_BDW)
SET(MORPHEUS_ARCH_AVX2 ON)
COMPILER_SPECIFIC_FLAGS(
  Intel   -xCORE-AVX2
#   PGI     -tp=haswell
  Cray    NO-VALUE-SPECIFIED
  DEFAULT -march=core-avx2 -mtune=core-avx2 -mrtm
)
ENDIF()

IF (MORPHEUS_ARCH_SKX)
#avx512-xeon
SET(MORPHEUS_ARCH_AVX512XEON ON)
COMPILER_SPECIFIC_FLAGS(
  Intel   -xCORE-AVX512
#   PGI     NO-VALUE-SPECIFIED
  Cray    NO-VALUE-SPECIFIED
  DEFAULT -march=skylake-avx512 -mtune=skylake-avx512 -mrtm
)
ENDIF()

IF (MORPHEUS_ARCH_WSM OR MORPHEUS_ARCH_SNB OR MORPHEUS_ARCH_HSW OR MORPHEUS_ARCH_BDW OR MORPHEUS_ARCH_SKX OR MORPHEUS_ARCH_ZEN OR MORPHEUS_ARCH_ZEN2 OR MORPHEUS_ARCH_ZEN3)
SET(MORPHEUS_USE_ISA_X86_64 ON)
ENDIF()

IF (MORPHEUS_ARCH_BDW OR MORPHEUS_ARCH_SKX)
SET(MORPHEUS_ENABLE_TM ON) #not a cache variable
ENDIF()

# IF (MORPHEUS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
# COMPILER_SPECIFIC_FLAGS(
#   Clang  -fcuda-rdc
#   NVIDIA --relocatable-device-code=true
# )
# ENDIF()

# Clang needs mcx16 option enabled for Windows atomic functions
IF (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND WIN32)
COMPILER_SPECIFIC_OPTIONS(
  Clang -mcx16
)
ENDIF()

# MSVC ABI has many deprecation warnings, so ignore them
IF (CMAKE_CXX_COMPILER_ID STREQUAL MSVC OR "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
COMPILER_SPECIFIC_DEFS(
  Clang _CRT_SECURE_NO_WARNINGS
)
ENDIF()


# #Right now we cannot get the compiler ID when cross-compiling, so just check
# #that HIP is enabled
# IF (MORPHEUS_ENABLE_HIP)
#     IF (MORPHEUS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
#     COMPILER_SPECIFIC_FLAGS(
#         DEFAULT -fgpu-rdc
#     )
#     ELSE()
#     COMPILER_SPECIFIC_FLAGS(
#         DEFAULT -fno-gpu-rdc
#     )
#     ENDIF()
# ENDIF()


# SET(CUDA_ARCH_ALREADY_SPECIFIED "")
# FUNCTION(CHECK_CUDA_ARCH ARCH FLAG)
# IF(MORPHEUS_ARCH_${ARCH})
#   IF(CUDA_ARCH_ALREADY_SPECIFIED)
#     MESSAGE(FATAL_ERROR "Multiple GPU architectures given! Already have ${CUDA_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again.")
#   ENDIF()
#   SET(CUDA_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
#   IF (NOT MORPHEUS_ENABLE_CUDA)
#     MESSAGE(WARNING "Given CUDA arch ${ARCH}, but Morpheus_ENABLE_CUDA is OFF. Option will be ignored.")
#     UNSET(MORPHEUS_ARCH_${ARCH} PARENT_SCOPE)
#   ELSE()
#     SET(MORPHEUS_CUDA_ARCH_FLAG ${FLAG} PARENT_SCOPE)
#     GLOBAL_APPEND(MORPHEUS_CUDA_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
#     IF(MORPHEUS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OR MORPHEUS_CXX_COMPILER_ID STREQUAL NVIDIA)
#       GLOBAL_APPEND(MORPHEUS_LINK_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
#     ENDIF()
#   ENDIF()
# ENDIF()
# LIST(APPEND MORPHEUS_CUDA_ARCH_FLAGS ${FLAG})
# SET(MORPHEUS_CUDA_ARCH_FLAGS ${MORPHEUS_CUDA_ARCH_FLAGS} PARENT_SCOPE)
# LIST(APPEND MORPHEUS_CUDA_ARCH_LIST ${ARCH})
# SET(MORPHEUS_CUDA_ARCH_LIST ${MORPHEUS_CUDA_ARCH_LIST} PARENT_SCOPE)
# ENDFUNCTION()


# #These will define MORPHEUS_CUDA_ARCH_FLAG
# #to the corresponding flag name if ON
# CHECK_CUDA_ARCH(PASCAL60  sm_60)
# CHECK_CUDA_ARCH(PASCAL61  sm_61)
# CHECK_CUDA_ARCH(VOLTA70   sm_70)
# CHECK_CUDA_ARCH(VOLTA72   sm_72)
# CHECK_CUDA_ARCH(TURING75  sm_75)
# CHECK_CUDA_ARCH(AMPERE80  sm_80)
# CHECK_CUDA_ARCH(AMPERE86  sm_86)

# SET(AMDGPU_ARCH_ALREADY_SPECIFIED "")
# FUNCTION(CHECK_AMDGPU_ARCH ARCH FLAG)
# IF(MORPHEUS_ARCH_${ARCH})
#   IF(AMDGPU_ARCH_ALREADY_SPECIFIED)
#     MESSAGE(FATAL_ERROR "Multiple GPU architectures given! Already have ${AMDGPU_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again.")
#   ENDIF()
#   SET(AMDGPU_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
#   IF (NOT MORPHEUS_ENABLE_HIP)
#     MESSAGE(WARNING "Given AMD GPU architecture ${ARCH}, but Morpheus_ENABLE_HIP is OFF. Option will be ignored.")
#     UNSET(MORPHEUS_ARCH_${ARCH} PARENT_SCOPE)
#   ELSE()
#     SET(MORPHEUS_AMDGPU_ARCH_FLAG ${FLAG} PARENT_SCOPE)
#     GLOBAL_APPEND(MORPHEUS_AMDGPU_OPTIONS "${AMDGPU_ARCH_FLAG}=${FLAG}")
#     IF(MORPHEUS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
#       GLOBAL_APPEND(MORPHEUS_LINK_OPTIONS "${AMDGPU_ARCH_FLAG}=${FLAG}")
#     ENDIF()
#   ENDIF()
# ENDIF()
# ENDFUNCTION()

# #These will define MORPHEUS_AMDGPU_ARCH_FLAG
# #to the corresponding flag name if ON
# CHECK_AMDGPU_ARCH(VEGA900 gfx900) # Radeon Instinct MI25
# CHECK_AMDGPU_ARCH(VEGA906 gfx906) # Radeon Instinct MI50 and MI60
# CHECK_AMDGPU_ARCH(VEGA908 gfx908)

# IF(MORPHEUS_ENABLE_HIP AND NOT AMDGPU_ARCH_ALREADY_SPECIFIED)
# MESSAGE(SEND_ERROR "HIP enabled but no AMD GPU architecture currently enabled. "
#                    "Please enable one AMD GPU architecture via -DMorpheus_ARCH_{..}=ON'.")
# ENDIF()


# IF(MORPHEUS_ENABLE_CUDA AND NOT CUDA_ARCH_ALREADY_SPECIFIED)
# # Try to autodetect the CUDA Compute Capability by asking the device
# SET(_BINARY_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/cmake/compile_tests/CUDAComputeCapabilityWorkdir)
# FILE(REMOVE_RECURSE ${_BINARY_TEST_DIR})
# FILE(MAKE_DIRECTORY ${_BINARY_TEST_DIR})

# TRY_RUN(
#   _RESULT
#   _COMPILE_RESULT
#   ${_BINARY_TEST_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc
#   COMPILE_DEFINITIONS -DSM_ONLY
#   RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY)

# # if user is using kokkos_compiler_launcher, above will fail.
# IF(NOT _COMPILE_RESULT OR NOT _RESULT EQUAL 0)
#   # check to see if CUDA is not already enabled (may happen when Morpheus is subproject)
#   GET_PROPERTY(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
#   # language has to be fully enabled, just checking for CMAKE_CUDA_COMPILER isn't enough
#   IF(NOT "CUDA" IN_LIST _ENABLED_LANGUAGES)
#     # make sure the user knows that we aren't using CUDA compiler for anything else
#     MESSAGE(STATUS "CUDA auto-detection of architecture failed with ${CMAKE_CXX_COMPILER}. Enabling CUDA language ONLY to auto-detect architecture...")
#     INCLUDE(CheckLanguage)
#     CHECK_LANGUAGE(CUDA)
#     IF(CMAKE_CUDA_COMPILER)
#       ENABLE_LANGUAGE(CUDA)
#     ELSE()
#       MESSAGE(STATUS "CUDA language could not be enabled")
#     ENDIF()
#   ENDIF()

#   # if CUDA was enabled, this will be defined
#   IF(CMAKE_CUDA_COMPILER)
#     # copy our test to .cu so cmake compiles as CUDA
#     CONFIGURE_FILE(
#       ${PROJECT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc
#       ${PROJECT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu
#       COPYONLY
#     )
#     # run test again
#     TRY_RUN(
#       _RESULT
#       _COMPILE_RESULT
#       ${_BINARY_TEST_DIR}
#       ${PROJECT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu
#       COMPILE_DEFINITIONS -DSM_ONLY
#       RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY)
#   ENDIF()
# ENDIF()

# LIST(FIND MORPHEUS_CUDA_ARCH_FLAGS sm_${_CUDA_COMPUTE_CAPABILITY} FLAG_INDEX)
# IF(_COMPILE_RESULT AND _RESULT EQUAL 0 AND NOT FLAG_INDEX EQUAL -1)
#   MESSAGE(STATUS "Detected CUDA Compute Capability ${_CUDA_COMPUTE_CAPABILITY}")
#   LIST(GET MORPHEUS_CUDA_ARCH_LIST ${FLAG_INDEX} ARCHITECTURE)
#   MORPHEUS_SET_OPTION(ARCH_${ARCHITECTURE} ON)
#   CHECK_CUDA_ARCH(${ARCHITECTURE} sm_${_CUDA_COMPUTE_CAPABILITY})
#   LIST(APPEND MORPHEUS_ENABLED_ARCH_LIST ${ARCHITECTURE})
# ELSE()
#   MESSAGE(SEND_ERROR "CUDA enabled but no NVIDIA GPU architecture currently enabled and auto-detection failed. "
#                      "Please give one -DMorpheus_ARCH_{..}=ON' to enable an NVIDIA GPU architecture.\n"
#                      "You can yourself try to compile ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc and run the executable. "
#                      "If you are cross-compiling, you should try to do this on a compute node.")
# ENDIF()
# ENDIF()

# IF (MORPHEUS_ENABLE_CUDA)
# #Regardless of version, make sure we define the general architecture name
# #Regardless of version, make sure we define the general architecture name
# IF (KOKKOS_ARCH_PASCAL60 OR KOKKOS_ARCH_PASCAL61)
#  SET(KOKKOS_ARCH_PASCAL ON)
# ENDIF()

# #Regardless of version, make sure we define the general architecture name
# IF (KOKKOS_ARCH_VOLTA70 OR KOKKOS_ARCH_VOLTA72)
#   SET(KOKKOS_ARCH_VOLTA ON)
# ENDIF()

# IF (KOKKOS_ARCH_AMPERE80 OR KOKKOS_ARCH_AMPERE86)
#   SET(KOKKOS_ARCH_AMPERE ON)
# ENDIF()
# ENDIF()

#CMake verbose is kind of pointless
#Let's just always print things
MESSAGE(STATUS "Built-in Execution Spaces:")

FOREACH (_BACKEND Cuda HIP)
STRING(TOUPPER ${_BACKEND} UC_BACKEND)
IF(MORPHEUS_ENABLE_${UC_BACKEND})
  IF(_DEVICE_PARALLEL)
    MESSAGE(FATAL_ERROR "Multiple device parallel execution spaces are not allowed! "
                        "Trying to enable execution space ${_BACKEND}, "
                        "but execution space ${_DEVICE_PARALLEL} is already enabled. "
                        "Remove the CMakeCache.txt file and re-configure.")
  ENDIF()
  IF (${_BACKEND} STREQUAL "Cuda")
    SET(_DEFAULT_DEVICE_MEMSPACE "Kokkos::${_BACKEND}Space")
    SET(_DEVICE_PARALLEL "Kokkos::${_BACKEND}")
  ELSE()
     SET(_DEFAULT_DEVICE_MEMSPACE "Kokkos::Experimental::${_BACKEND}Space")
     SET(_DEVICE_PARALLEL "Kokkos::Experimental::${_BACKEND}")
  ENDIF()
ENDIF()
ENDFOREACH()
IF(NOT _DEVICE_PARALLEL)
SET(_DEVICE_PARALLEL "NoTypeDefined")
SET(_DEFAULT_DEVICE_MEMSPACE "NoTypeDefined")
ENDIF()
MESSAGE(STATUS "    Device Parallel: ${_DEVICE_PARALLEL}")

FOREACH (_BACKEND OpenMP)
STRING(TOUPPER ${_BACKEND} UC_BACKEND)
IF(MORPHEUS_ENABLE_${UC_BACKEND})
  IF(_HOST_PARALLEL)
    MESSAGE(FATAL_ERROR "Multiple host parallel execution spaces are not allowed! "
                        "Trying to enable execution space ${_BACKEND}, "
                        "but execution space ${_HOST_PARALLEL} is already enabled. "
                        "Remove the CMakeCache.txt file and re-configure.")
  ENDIF()
  SET(_HOST_PARALLEL "Kokkos::${_BACKEND}")
ENDIF()
ENDFOREACH()

IF(NOT _HOST_PARALLEL AND NOT MORPHEUS_ENABLE_SERIAL)
MESSAGE(FATAL_ERROR "At least one host execution space must be enabled, "
                    "but no host parallel execution space was requested "
                    "and Morpheus_ENABLE_SERIAL=OFF.")
ENDIF()

IF(_HOST_PARALLEL)
MESSAGE(STATUS "    Host Parallel: ${_HOST_PARALLEL}")
ELSE()
SET(_HOST_PARALLEL "NoTypeDefined")
MESSAGE(STATUS "    Host Parallel: NoTypeDefined")
ENDIF()

IF(MORPHEUS_ENABLE_SERIAL)
MESSAGE(STATUS "      Host Serial: SERIAL")
ELSE()
MESSAGE(STATUS "      Host Serial: NONE")
ENDIF()

MESSAGE(STATUS "")
MESSAGE(STATUS "Architectures:")
FOREACH(Arch ${MORPHEUS_ENABLED_ARCH_LIST})
MESSAGE(STATUS " ${Arch}")
ENDFOREACH()