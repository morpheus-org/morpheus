# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
  message( FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files." )
endif()

# Basic initialization (Used in MORPHEUS_SETTINGS)
SET(Morpheus_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
SET(MORPHEUS_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
SET(MORPHEUS_SRC_PATH      ${Morpheus_SOURCE_DIR})
SET(MORPHEUS_PATH          ${Morpheus_SOURCE_DIR})
SET(MORPHEUS_TOP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Needed to simplify syntax of if statements
CMAKE_POLICY(SET CMP0054 NEW)
# Needed to make IN_LIST a valid operator
CMAKE_POLICY(SET CMP0057 NEW)

# Is this build a subdirectory of another project
GET_DIRECTORY_PROPERTY(HAS_PARENT PARENT_DIRECTORY)

INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_functions.cmake)
INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_pick_cxx_std.cmake)

# SET(KOKKOS_ENABLED_OPTIONS)      #exported in config file
# SET(KOKKOS_ENABLED_DEVICES)      #exported in config file
# SET(KOKKOS_ENABLED_TPLS)         #exported in config file
# SET(KOKKOS_ENABLED_ARCH_LIST)    #exported in config file

# #These are helper flags used for sanity checks during config
# #Certain features should depend on other features being configured first
# SET(KOKKOS_CFG_DAG_NONE              On) #sentinel to indicate no dependencies
# SET(KOKKOS_CFG_DAG_DEVICES_DONE      Off)
# SET(KOKKOS_CFG_DAG_OPTIONS_DONE      Off)
# SET(KOKKOS_CFG_DAG_ARCH_DONE         Off)
# SET(KOKKOS_CFG_DAG_CXX_STD_DONE      Off)
# SET(KOKKOS_CFG_DAG_COMPILER_ID_DONE  Off)
# FUNCTION(KOKKOS_CFG_DEPENDS SUCCESSOR PRECURSOR)
#   SET(PRE_FLAG  KOKKOS_CFG_DAG_${PRECURSOR})
#   SET(POST_FLAG KOKKOS_CFG_DAG_${SUCCESSOR})
#   IF (NOT ${PRE_FLAG})
#     MESSAGE(FATAL_ERROR "Bad CMake refactor: feature ${SUCCESSOR} cannot be configured until ${PRECURSOR} is configured")
#   ENDIF()
#   GLOBAL_SET(${POST_FLAG} On)
# ENDFUNCTION()


LIST(APPEND CMAKE_MODULE_PATH cmake/Modules)

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Always call the project command to define Morpheus_ variables
# and to make sure that C++ is an enabled language
PROJECT(Morpheus CXX)
IF(NOT HAS_PARENT)
  IF (NOT CMAKE_BUILD_TYPE)
    SET(DEFAULT_BUILD_TYPE "RelWithDebInfo")
    MESSAGE(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    SET(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
        FORCE)
  ENDIF()
ENDIF()


IF (NOT CMAKE_SIZEOF_VOID_P)
  STRING(FIND ${CMAKE_CXX_COMPILER} nvcc_wrapper FIND_IDX)
  IF (NOT FIND_IDX STREQUAL -1)
    MESSAGE(FATAL_ERROR "Morpheus did not configure correctly and failed to validate compiler. The most likely cause is CUDA linkage using nvcc_wrapper. Please ensure your CUDA environment is correctly configured.")
  ELSE()
    MESSAGE(FATAL_ERROR "Morpheus did not configure correctly and failed to validate compiler. The most likely cause is linkage errors during CMake compiler validation. Please consult the CMake error log shown below for the exact error during compiler validation")
  ENDIF()
ELSEIF (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  MESSAGE(FATAL_ERROR "Morpheus assumes a 64-bit build; i.e., 8-byte pointers, but found ${CMAKE_SIZEOF_VOID_P}-byte pointers instead")
ENDIF()


set(Morpheus_VERSION_MAJOR 0)
set(Morpheus_VERSION_MINOR 2)
set(Morpheus_VERSION_PATCH 00)
set(Morpheus_VERSION "${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR}.${Morpheus_VERSION_PATCH}")
math(EXPR Morpheus_VERSION "${Morpheus_VERSION_MAJOR} * 10000 + ${Morpheus_VERSION_MINOR} * 100 + ${Morpheus_VERSION_PATCH}")

MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
CMAKE_POLICY(SET CMP0074 NEW)

# Load either the real TriBITS or a TriBITS wrapper
# for certain utility functions that are universal (like GLOBAL_SET)
INCLUDE(${MORPHEUS_SRC_PATH}/cmake/fake_tribits.cmake)
MESSAGE(STATUS "${PROJECT_NAME} version is: v${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR}.${Morpheus_VERSION_PATCH} (${Morpheus_VERSION})")

IF (Morpheus_ENABLE_CUDA)
  # If we are building CUDA, we have tricked CMake because we declare a CXX project
  # If the default C++ standard for a given compiler matches the requested
  # standard, then CMake just omits the -std flag in later versions of CMake
  # This breaks CUDA compilation (CUDA compiler can have a different default
  # -std then the underlying host compiler by itself). Setting this variable
  # forces CMake to always add the -std flag even if it thinks it doesn't need it
  GLOBAL_SET(CMAKE_CXX_STANDARD_DEFAULT 98)
ENDIF()

# These are the variables we will append to as we go
# I really wish these were regular variables
# but scoping issues can make it difficult
GLOBAL_SET(MORPHEUS_COMPILE_OPTIONS)
GLOBAL_SET(MORPHEUS_LINK_OPTIONS)
GLOBAL_SET(MORPHEUS_CUDA_OPTIONS)
GLOBAL_SET(MORPHEUS_CUDAFE_OPTIONS)
GLOBAL_SET(MORPHEUS_XCOMPILER_OPTIONS)
# We need to append text here for making sure TPLs
# we import are available for an installed Morpheus
GLOBAL_SET(MORPHEUS_TPL_EXPORTS)
# MORPHEUS_DEPENDENCE is used by morpheus_launch_compiler
GLOBAL_SET(MORPHEUS_COMPILE_DEFINITIONS MORPHEUS_DEPENDENCE)
# MSVC never goes through morpheus_launch_compiler
IF(NOT MSVC)
    GLOBAL_APPEND(MORPHEUS_LINK_OPTIONS -DMORPHEUS_DEPENDENCE)
ENDIF()

# Include a set of Morpheus-specific wrapper functions that
# will either call raw CMake or TriBITS
# These are functions like MORPHEUS_INCLUDE_DIRECTORIES
INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_tribits.cmake)


# Check the environment and set certain variables
# to allow platform-specific checks
INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_check_env.cmake)

# The build environment setup goes in the following steps
# 1) Check all the enable options. This includes checking MORPHEUS_DEVICES
# 2) Check the compiler ID (type and version)
# 3) Check the CXX standard and select important CXX flags
# 4) Check for any third-party libraries (TPLs) like hwloc
# 5) Check if optimizing for a particular architecture and add arch-specific flags
MORPHEUS_SETUP_BUILD_ENVIRONMENT()

# Finish off the build
# 6) Recurse into subdirectories and configure individual libraries
# 7) Export and install targets

OPTION(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# Workaround for building position independent code.
IF(BUILD_SHARED_LIBS)
  SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

SET(MORPHEUS_EXT_LIBRARIES Morpheus::morpheus Morpheus::morpheuscore)
SET(MORPHEUS_INT_LIBRARIES morpheus morpheuscore)
SET_PROPERTY(GLOBAL PROPERTY MORPHEUS_INT_LIBRARIES ${MORPHEUS_INT_LIBRARIES})

IF(HAS_PARENT)
  SET(MORPHEUS_HEADER_DIR "include/morpheus")
  SET(MORPHEUS_IS_SUBDIRECTORY TRUE)
ELSE()
  SET(MORPHEUS_HEADER_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
  SET(MORPHEUS_IS_SUBDIRECTORY FALSE)
ENDIF()


#------------------------------------------------------------------------------
#
# A) Forward declare the package so that certain options are also defined for
# subpackages

MORPHEUS_PACKAGE_DECL()

#------------------------------------------------------------------------------
#
# D) Process the subpackages (subdirectories) for Morpheus
#
MORPHEUS_PROCESS_SUBPACKAGES()

#------------------------------------------------------------------------------
#
# E) If Morpheus itself is enabled, process the Morpheus package
#

MORPHEUS_PACKAGE_DEF()
MORPHEUS_EXCLUDE_AUTOTOOLS_FILES()
MORPHEUS_PACKAGE_POSTPROCESS()
MORPHEUS_CONFIGURE_CORE()

IF (NOT Kokkos_INSTALL_TESTING)
  ADD_LIBRARY(morpheus INTERFACE)
  #Make sure in-tree projects can reference this as Morpheus::
  #to match the installed target names
  ADD_LIBRARY(Morpheus::morpheus ALIAS morpheus)
  TARGET_LINK_LIBRARIES(morpheus INTERFACE morpheuscore)
  KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL(morpheus)
ENDIF()
INCLUDE(${KOKKOS_SRC_PATH}/cmake/morpheus_install.cmake)

# nvcc_wrapper is Kokkos' wrapper for NVIDIA's NVCC CUDA compiler.
# Morpheus needs nvcc_wrapper in order to build.  Other libraries and
# executables also need nvcc_wrapper.  Thus, we need to install it.
# If the argument of DESTINATION is a relative path, CMake computes it
# as relative to ${CMAKE_INSTALL_PATH}.
# MORPHEUS_INSTALL_ADDITIONAL_FILES will install nvcc wrapper and other generated
# files
MORPHEUS_INSTALL_ADDITIONAL_FILES()


#  Finally - if we are a subproject - make sure the enabled devices are visible
IF (HAS_PARENT)
  FOREACH(DEV Morpheus_ENABLED_DEVICES)
    #I would much rather not make these cache variables or global properties, but I can't
    #make any guarantees on whether PARENT_SCOPE is good enough to make
    #these variables visible where I need them
    SET(Morpheus_ENABLE_${DEV} ON PARENT_SCOPE)
    SET_PROPERTY(GLOBAL PROPERTY Morpheus_ENABLE_${DEV} ON)
  ENDFOREACH()
ENDIF()

# include(cmake/Modules/MorpheusVersion.cmake)
# include(cmake/Modules/MorpheusDependenciesConfigure.cmake)

# enable_language(CXX)

# # Morpheus Directories
# set(Morpheus_DIR ${Morpheus_SOURCE_DIR})
# set(Morpheus_INCLUDE_DIR ${Morpheus_SOURCE_DIR}/morpheus)
# set(Morpheus_EXAMPLES_DIR ${Morpheus_SOURCE_DIR}/examples)
# set(Morpheus_BENCHMARKS_DIR ${Morpheus_SOURCE_DIR}/benchmarks)

# # Write morpheus version
# write_version()

# # Set build type
# set(Morpheus_DEFAULT_BUILD_TYPE "Release")

# if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#   message(STATUS "Morpheus: Setting build type to '${Morpheus_DEFAULT_BUILD_TYPE}' since none specified.")
#   set(CMAKE_BUILD_TYPE
#       "${Morpheus_DEFAULT_BUILD_TYPE}"
#       CACHE STRING "Choose the type of build." FORCE)
#   # Set the possible values of build type for cmake-gui
#   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
#                                                "RelWithDebInfo")
# endif()

# IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
#   MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
#   CMAKE_POLICY(SET CMP0074 NEW)
# ENDIF()

# # Build options
# option(Morpheus_BUILD_TESTS "Configure CMake to build unit tests using CATCH2" OFF)
# option(Morpheus_BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
# option(Morpheus_BUILD_EXAMPLES "Configure CMake to build examples" OFF)
# option(Morpheus_INSTALL "Enable CMake install commands" ON)

# # Source files
# add_subdirectory(morpheus)

# # Build Tests
# if(Morpheus_BUILD_TESTS)
#   # enable_testing()
#   add_subdirectory(tests)
# endif()

# # Build Examples
# if(Morpheus_BUILD_EXAMPLES)
#   add_subdirectory(examples)
# endif()

# # Build Benchmarks
# if(Morpheus_BUILD_BENCHMARKS)
#   add_subdirectory(benchmarks)
# endif()

# # Make Documentation
# add_custom_command(
#   OUTPUT Morpheus_DOXYGEN
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
#   COMMAND doxygen Doxyfile
#   VERBATIM
#   COMMENT "Custom command for Morpheus doxygen docs")

# add_custom_target(
#   morpheus_doc
#   DEPENDS Morpheus_DOXYGEN
#   COMMENT "Target for the custom command to build the Morpheus doxygen docs")