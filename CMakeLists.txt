# Disable in-source builds to prevent source tree corruption.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(
    FATAL_ERROR
      "FATAL: In-source builds are not allowed. You should create a separate directory for build files."
  )
endif()

# ~~~
# We want to determine if options are given with the wrong case.
# In order to detect which arguments are given to compare against the list of
# valid arguments, at the beginning here we need to form a list of all the given variables.
# If it begins with any case of MoRpHEUs, we add it to the list.
# ~~~
get_cmake_property(_variableNames VARIABLES)
set(MORPHEUS_GIVEN_VARIABLES)
foreach(var ${_variableNames})
  string(TOUPPER ${var} UC_VAR)
  string(FIND ${UC_VAR} MORPHEUS IDX)
  if(${IDX} EQUAL 0)
    list(APPEND MORPHEUS_GIVEN_VARIABLES ${var})
  endif()
endforeach()

# Basic initialization (Used in MORPHEUS_SETTINGS)
set(Morpheus_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MORPHEUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MORPHEUS_SRC_PATH ${Morpheus_SOURCE_DIR})
set(MORPHEUS_PATH ${Morpheus_SOURCE_DIR})
set(MORPHEUS_TOP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MORPHEUS_TOP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Needed to simplify syntax of if statements
cmake_policy(SET CMP0054 NEW)
# Needed to make IN_LIST a valid operator
cmake_policy(SET CMP0057 NEW)

# Is this build a subdirectory of another project
get_directory_property(MORPHEUS_HAS_PARENT PARENT_DIRECTORY)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Always call the project command to define Morpheus_ variables and to make sure
# that C++ is an enabled language
project(Morpheus CXX)
if(NOT MOPRHEUS_HAS_PARENT)
  if(NOT CMAKE_BUILD_TYPE)
    set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
    message(
      STATUS
        "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "${DEFAULT_BUILD_TYPE}"
        CACHE
          STRING
          "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
          FORCE)
  endif()
endif()

if(NOT CMAKE_SIZEOF_VOID_P)
  string(FIND ${CMAKE_CXX_COMPILER} nvcc_wrapper FIND_IDX)
  if(NOT FIND_IDX STREQUAL -1)
    message(
      FATAL_ERROR
        "Morpheus did not configure correctly and failed to validate compiler. The most likely cause is CUDA linkage using nvcc_wrapper. Please ensure your CUDA environment is correctly configured."
    )
  else()
    message(
      FATAL_ERROR
        "Morpheus did not configure correctly and failed to validate compiler. The most likely cause is linkage errors during CMake compiler validation. Please consult the CMake error log shown below for the exact error during compiler validation"
    )
  endif()
elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(
    FATAL_ERROR
      "Morpheus assumes a 64-bit build; i.e., 8-byte pointers, but found ${CMAKE_SIZEOF_VOID_P}-byte pointers instead"
  )
endif()

set(Morpheus_VERSION_MAJOR 0)
set(Morpheus_VERSION_MINOR 03)
set(Morpheus_VERSION_PATCH 00)
set(Morpheus_VERSION
    "${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR}.${Morpheus_VERSION_PATCH}"
)
math(
  EXPR
  MORPHEUS_VERSION
  "${Morpheus_VERSION_MAJOR} * 10000 + ${Morpheus_VERSION_MINOR} * 100 + ${Morpheus_VERSION_PATCH}"
)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

message(
  STATUS
    "${PROJECT_NAME} version is: v${Morpheus_VERSION} (${MORPHEUS_VERSION})")

# load utility functions
include(cmake/morpheus_utils.cmake)

# Place here variables we will append as we go
global_set(MORPHEUS_TPL_EXPORTS)

# The build environment setup goes in the following steps
#
# 1.
morpheus_setup_build_environment() # TODO

# Finish off the build
#
# 1. Recurse into subdirectories and configure individual libraries
# 2. Export and install targets

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# Workaround for building position independent code.
if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(MORPHEUS_EXT_LIBRARIES Morpheus::morpheus Morpheus::morpheuscore)
set(MORPHEUS_INT_LIBRARIES morpheus morpheuscore)
set_property(GLOBAL PROPERTY MORPHEUS_INT_LIBRARIES ${MORPHEUS_INT_LIBRARIES})

include(GNUInstallDirs)
if(MORPHEUS_HAS_PARENT)
  set(MORPHEUS_HEADER_INSTALL_DIR "include/morpheus")
  set(MORPHEUS_IS_SUBDIRECTORY TRUE)
else()
  set(MORPHEUS_HEADER_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
  set(MORPHEUS_IS_SUBDIRECTORY FALSE)
endif()

# Forward declare the package for certain options to be defined for subpackages
morpheus_package_decl()

# Process the subpackages (subdirectories) for Morpheus
morpheus_process_subpackages()

# If Morpheus itself is enabled, process the Morpheus package
if(NOT Morpheus_INSTALL_TESTING)
  add_library(morpheus INTERFACE)
  # ~~~
  # Make sure in-tree projects can reference this as Morpheus::
  # to match the installed target names
  # ~~~
  add_library(Morpheus::morpheus ALIAS morpheus)
  target_link_libraries(morpheus INTERFACE morpheuscore)
  morpheus_internal_add_library_install(morpheus)
endif()

morpheus_package_postprocess()
