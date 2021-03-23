cmake_minimum_required(VERSION 3.12)

project(Morpheus
        VERSION 0.1.0)

include(cmake/Modules/MorpheusVersion.cmake)
include(cmake/Modules/MorpheusDependenciesConfigure.cmake)

enable_language(CXX)

# Morpheus Directories
set(Morpheus_DIR ${Morpheus_SOURCE_DIR})
set(Morpheus_INCLUDE_DIR ${Morpheus_SOURCE_DIR}/morpheus)
set(Morpheus_EXAMPLES_DIR ${Morpheus_SOURCE_DIR}/examples)

# Write morpheus version
write_version()

# Set build type
set(Morpheus_DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Morpheus: Setting build type to '${Morpheus_DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE
      "${Morpheus_DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

# Build options
option(Morpheus_BUILD_TESTS "Configure CMake to build unit tests using CATCH2" OFF)
option(Morpheus_BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
option(Morpheus_BUILD_EXAMPLES "Configure CMake to build examples" OFF)
option(Morpheus_INSTALL "Enable CMake install commands" ON)

# Source files
add_subdirectory(morpheus)

# Build Tests
if(Morpheus_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Build Examples
if(Morpheus_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build Benchmarks
if(Morpheus_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Make Documentation
add_custom_command(
  OUTPUT Morpheus_DOXYGEN
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for Morpheus doxygen docs")

add_custom_target(
  morpheus_doc
  DEPENDS Morpheus_DOXYGEN
  COMMENT "Target for the custom command to build the Morpheus doxygen docs")