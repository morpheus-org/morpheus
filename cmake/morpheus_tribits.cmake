#These are tribits wrappers only ever called by Morpheus itself

INCLUDE(CMakeParseArguments)
INCLUDE(CTest)
INCLUDE(GNUInstallDirs)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

FUNCTION(VERIFY_EMPTY CONTEXT)
  if(${ARGN})
    MESSAGE(FATAL_ERROR "Morpheus does not support all of Tribits. Unhandled arguments in ${CONTEXT}:\n${ARGN}")
  endif()
ENDFUNCTION()

MACRO(MORPHEUS_SETUP_BUILD_ENVIRONMENT)
  # This is needed for both regular build and install tests
  INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_compiler_id.cmake)
  #set an internal option, if not already set
  SET(Morpheus_INSTALL_TESTING OFF CACHE INTERNAL "Whether to build tests and examples against installation")
  IF (Morpheus_INSTALL_TESTING)
    SET(MORPHEUS_ENABLE_TESTS ON)
    SET(MORPHEUS_ENABLE_EXAMPLES ON)
    # This looks a little weird, but what we are doing
    # is to NOT build Morpheus but instead look for an
    # installed Morpheus - then build examples and tests
    # against that installed Morpheus
    FIND_PACKAGE(Morpheus REQUIRED)
    # Just grab the configuration from the installation
    FOREACH(DEV ${Morpheus_DEVICES})
      SET(MORPHEUS_ENABLE_${DEV} ON)
    ENDFOREACH()
    FOREACH(OPT ${Morpheus_OPTIONS})
      SET(MORPHEUS_ENABLE_${OPT} ON)
    ENDFOREACH()
    FOREACH(TPL ${Morpheus_TPLS})
      SET(MORPHEUS_ENABLE_${TPL} ON)
    ENDFOREACH()
    FOREACH(ARCH ${Morpheus_ARCH})
      SET(MORPHEUS_ARCH_${ARCH} ON)
    ENDFOREACH()
  ELSE()
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_enable_devices.cmake)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_enable_options.cmake)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_test_cxx_std.cmake)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_arch.cmake)
    SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Morpheus_SOURCE_DIR}/cmake/Modules/")
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_tpls.cmake)
    INCLUDE(${MORPHEUS_SRC_PATH}/cmake/morpheus_corner_cases.cmake)
  ENDIF()
ENDMACRO()

MACRO(MORPHEUS_PACKAGE_DECL)

  SET(PACKAGE_NAME Morpheus)
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)

ENDMACRO()

MACRO(MORPHEUS_PROCESS_SUBPACKAGES)

  ADD_SUBDIRECTORY(core)
  ADD_SUBDIRECTORY(examples)

ENDMACRO()

MACRO(MORPHEUS_PACKAGE_DEF)
  #do nothing
ENDMACRO()

MACRO(MORPHEUS_PACKAGE_POSTPROCESS)
  #do nothing
ENDMACRO()

## MORPHUES_CONFIGURE_CORE  Configure/Generate header files for core content based
##                        on enabled backends.
##                        MORPHEUS_FWD is the forward declare set
##                        MORPHEUS_SETUP  is included in Morpheus_Macros.hpp and include prefix includes/defines
##                        MORPHEUS_DECLARE is the declaration set
##                        MORPHEUS_POST_INCLUDE is included at the end of Morpheus_Core.hpp
MACRO(MORPHEUS_CONFIGURE_CORE)
   SET(FWD_BACKEND_LIST)
   FOREACH(MEMSPACE ${MORPHEUS_MEMSPACE_LIST})
      LIST(APPEND FWD_BACKEND_LIST ${MEMSPACE})
   ENDFOREACH()
   FOREACH(BACKEND_ ${MORPHEUS_ENABLED_DEVICES})
      IF( ${BACKEND_} STREQUAL "PTHREAD")
         LIST(APPEND FWD_BACKEND_LIST THREADS)
      ELSE()
         LIST(APPEND FWD_BACKEND_LIST ${BACKEND_})
      ENDIF()
   ENDFOREACH()
   MESSAGE(STATUS "Morpheus Devices: ${MORPHEUS_ENABLED_DEVICES}, Morpheus Backends: ${FWD_BACKEND_LIST}")
   MORPHEUS_CONFIG_HEADER( MorpheusCore_Config_HeaderSet.in MorpheusCore_Config_FwdBackend.hpp "MORPHEUS_FWD" "fwd/Morpheus_Fwd" "${FWD_BACKEND_LIST}")
   MORPHEUS_CONFIG_HEADER( MorpheusCore_Config_HeaderSet.in MorpheusCore_Config_SetupBackend.hpp "MORPHEUS_SETUP" "setup/Morpheus_Setup" "${DEVICE_SETUP_LIST}")
   MORPHEUS_CONFIG_HEADER( MorpheusCore_Config_HeaderSet.in MorpheusCore_Config_DeclareBackend.hpp "MORPHEUS_DECLARE" "decl/Morpheus_Declare" "${FWD_BACKEND_LIST}")
   MORPHEUS_CONFIG_HEADER( MorpheusCore_Config_HeaderSet.in MorpheusCore_Config_PostInclude.hpp "MORPHEUS_POST_INCLUDE" "Morpheus_Post_Include" "${MORPHEUS_BACKEND_POST_INCLUDE_LIST}")
   SET(_DEFAULT_HOST_MEMSPACE "::Morpheus::HostSpace")
   MORPHEUS_OPTION(DEFAULT_DEVICE_MEMORY_SPACE "" STRING "Override default device memory space")
   MORPHEUS_OPTION(DEFAULT_HOST_MEMORY_SPACE "" STRING "Override default host memory space")
   MORPHEUS_OPTION(DEFAULT_DEVICE_EXECUTION_SPACE "" STRING "Override default device execution space")
   MORPHEUS_OPTION(DEFAULT_HOST_PARALLEL_EXECUTION_SPACE "" STRING "Override default host parallel execution space")
   IF (NOT Morpheus_DEFAULT_DEVICE_EXECUTION_SPACE STREQUAL "")
      SET(_DEVICE_PARALLEL ${Morpheus_DEFAULT_DEVICE_EXECUTION_SPACE})
      MESSAGE(STATUS "Override default device execution space: ${_DEVICE_PARALLEL}")
      SET(MORPHEUS_DEVICE_SPACE_ACTIVE ON)
   ELSE()
      IF (_DEVICE_PARALLEL STREQUAL "NoTypeDefined")
         SET(MORPHEUS_DEVICE_SPACE_ACTIVE OFF)
      ELSE()
         SET(MORPHEUS_DEVICE_SPACE_ACTIVE ON)
      ENDIF()
   ENDIF()
   IF (NOT Morpheus_DEFAULT_HOST_PARALLEL_EXECUTION_SPACE STREQUAL "")
      SET(_HOST_PARALLEL ${Morpheus_DEFAULT_HOST_PARALLEL_EXECUTION_SPACE})
      MESSAGE(STATUS "Override default host parallel execution space: ${_HOST_PARALLEL}")
      SET(MORPHEUS_HOSTPARALLEL_SPACE_ACTIVE ON)
   ELSE()
      IF (_HOST_PARALLEL STREQUAL "NoTypeDefined")
         SET(MORPHEUS_HOSTPARALLEL_SPACE_ACTIVE OFF)
      ELSE()
         SET(MORPHEUS_HOSTPARALLEL_SPACE_ACTIVE ON)
      ENDIF()
   ENDIF()
   #We are ready to configure the header
   CONFIGURE_FILE(cmake/MorpheusCore_config.hpp.in MoprheusCore_config.hpp @ONLY)
ENDMACRO()

MACRO(MORPHEUS_INTERNAL_ADD_LIBRARY_INSTALL LIBRARY_NAME)
  MORPHEUS_LIB_TYPE(${LIBRARY_NAME} INCTYPE)
  TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} ${INCTYPE} $<INSTALL_INTERFACE:${MORPHEUS_HEADER_DIR}>)

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ${PACKAGE_NAME}
  )

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT MorpheusTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  VERIFY_EMPTY(MORPHEUS_ADD_LIBRARY ${PARSE_UNPARSED_ARGUMENTS})
ENDMACRO()

## MORPHEUS_INSTALL_ADDITIONAL_FILES - instruct cmake to install files in target destination.
##                        Includes generated header files, scripts such as nvcc_wrapper and hpcbind,
##                        as well as other files provided through plugins.
MACRO(MORPHEUS_INSTALL_ADDITIONAL_FILES)

  # # kokkos_launch_compiler is used by Kokkos to prefix compiler commands so that they forward to original kokkos compiler
  # # if nvcc_wrapper was not used as CMAKE_CXX_COMPILER, configure the original compiler into kokkos_launch_compiler
  # IF(NOT "${CMAKE_CXX_COMPILER}" MATCHES "nvcc_wrapper")
  #   SET(NVCC_WRAPPER_DEFAULT_COMPILER "${CMAKE_CXX_COMPILER}")
  # ELSE()
  #   IF(NOT "$ENV{NVCC_WRAPPER_DEFAULT_COMPILER}" STREQUAL "")
  #       SET(NVCC_WRAPPER_DEFAULT_COMPILER "$ENV{NVCC_WRAPPER_DEFAULT_COMPILER}")
  #   ENDIF()
  # ENDIF()

  # CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/bin/kokkos_launch_compiler
  #   ${PROJECT_BINARY_DIR}/temp/kokkos_launch_compiler
  #   @ONLY)

  # INSTALL(PROGRAMS
  #         "${CMAKE_CURRENT_SOURCE_DIR}/bin/nvcc_wrapper"
  #         "${CMAKE_CURRENT_SOURCE_DIR}/bin/hpcbind"
  #         "${CMAKE_CURRENT_SOURCE_DIR}/bin/kokkos_launch_compiler"
  #         "${PROJECT_BINARY_DIR}/temp/kokkos_launch_compiler"
  #         DESTINATION ${CMAKE_INSTALL_BINDIR})
  INSTALL(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_config.h"
          "${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_Config_FwdBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_Config_SetupBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_Config_DeclareBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_Config_PostInclude.hpp"
          DESTINATION ${KOKKOS_HEADER_DIR})
ENDMACRO()

MACRO(MORPHEUS_SUBPACKAGE NAME)
  SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  SET(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
  SET(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  #ADD_INTERFACE_LIBRARY(PACKAGE_${PACKAGE_NAME})
  #GLOBAL_SET(${PACKAGE_NAME}_LIBS "")
ENDMACRO()

MACRO(MORPHEUS_SUBPACKAGE_POSTPROCESS)
  # do nothing
ENDMACRO()

MACRO(MORPHEUS_ADD_TEST_DIRECTORIES)
  IF(MORPHEUS_ENABLE_TESTS)
    FOREACH(TEST_DIR ${ARGN})
      ADD_SUBDIRECTORY(${TEST_DIR})
    ENDFOREACH()
  ENDIF()
ENDMACRO()

MACRO(MORPHEUS_ADD_EXAMPLE_DIRECTORIES)
  IF(MORPHEUS_ENABLE_EXAMPLES)
    FOREACH(EXAMPLE_DIR ${ARGN})
      ADD_SUBDIRECTORY(${EXAMPLE_DIR})
    ENDFOREACH()
  ENDIF()
ENDMACRO()

FUNCTION(MORPHEUS_ADD_LIBRARY LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    "ADD_BUILD_OPTIONS"
    ""
    "HEADERS"
    ${ARGN}
  )
  
  # Forward the headers, we want to know about all headers
  # to make sure they appear correctly in IDEs
  MORPHEUS_INTERNAL_ADD_LIBRARY(
    ${LIBRARY_NAME} ${PARSE_UNPARSED_ARGUMENTS} HEADERS ${PARSE_HEADERS})
  IF (PARSE_ADD_BUILD_OPTIONS)
    MORPHEUS_SET_LIBRARY_PROPERTIES(${LIBRARY_NAME})
  ENDIF()
ENDFUNCTION()

FUNCTION(MORPHEUS_INTERNAL_ADD_LIBRARY LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    "STATIC;SHARED"
    ""
    "HEADERS;SOURCES"
    ${ARGN})

  IF(PARSE_HEADERS)
    LIST(REMOVE_DUPLICATES PARSE_HEADERS)
  ENDIF()
  IF(PARSE_SOURCES)
    LIST(REMOVE_DUPLICATES PARSE_SOURCES)
  ENDIF()

  IF(PARSE_STATIC)
    SET(LINK_TYPE STATIC)
  ENDIF()

  IF(PARSE_SHARED)
    SET(LINK_TYPE SHARED)
  ENDIF()

  # MSVC and other platforms want to have
  # the headers included as source files
  # for better dependency detection
  ADD_LIBRARY(
    ${LIBRARY_NAME}
    ${LINK_TYPE}
    ${PARSE_HEADERS}
    ${PARSE_SOURCES}
  )

  IF(PARSE_SHARED OR BUILD_SHARED_LIBS)
    SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES
      VERSION   ${Morpheus_VERSION}
      SOVERSION ${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR}
    )
  ENDIF()

  MORPHEUS_INTERNAL_ADD_LIBRARY_INSTALL(${LIBRARY_NAME})

  #In case we are building in-tree, add an alias name
  #that matches the install Morpheus:: name
  ADD_LIBRARY(Morpheus::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})
ENDFUNCTION()

FUNCTION(MORPHEUS_SET_LIBRARY_PROPERTIES LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    "PLAIN_STYLE"
    ""
    ""
    ${ARGN})

  IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
    #I can use link options
    #check for CXX linkage using the simple 3.18 way
    TARGET_LINK_OPTIONS(
      ${LIBRARY_NAME} PUBLIC
      $<$<LINK_LANGUAGE:CXX>:${MORPHEUS_LINK_OPTIONS}>
    )
  ELSE()
    #I can use link options
    #just assume CXX linkage
    TARGET_LINK_OPTIONS(
      ${LIBRARY_NAME} PUBLIC ${MORPHEUS_LINK_OPTIONS}
    )
  ENDIF()

  TARGET_COMPILE_OPTIONS(
    ${LIBRARY_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${MORPHEUS_COMPILE_OPTIONS}>
  )

  TARGET_COMPILE_DEFINITIONS(
    ${LIBRARY_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${MORPHEUS_COMPILE_DEFINITIONS}>
  )

  TARGET_LINK_LIBRARIES(
    ${LIBRARY_NAME} PUBLIC ${MORPHEUS_LINK_LIBRARIES}
  )

  IF (MORPHEUS_ENABLE_CUDA)
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MORPHEUS_CUDA_OPTIONS}>
    )
    SET(NODEDUP_CUDAFE_OPTIONS)
    FOREACH(OPT ${MORPHEUS_CUDAFE_OPTIONS})
      LIST(APPEND NODEDUP_CUDAFE_OPTIONS -Xcudafe ${OPT})
    ENDFOREACH()
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_CUDAFE_OPTIONS}>
    )
  ENDIF()

  LIST(LENGTH MORPHEUS_XCOMPILER_OPTIONS XOPT_LENGTH)
  IF (XOPT_LENGTH GREATER 1)
    MESSAGE(FATAL_ERROR "CMake deduplication does not allow multiple -Xcompiler flags (${KOKKOS_XCOMPILER_OPTIONS}): will require Morpheus to upgrade to minimum 3.12")
  ENDIF()
  IF(MORPHEUS_XCOMPILER_OPTIONS)
    SET(NODEDUP_XCOMPILER_OPTIONS)
    FOREACH(OPT ${MORPHEUSXCOMPILER_OPTIONS})
      #I have to do this for now because we can't guarantee 3.12 support
      #I really should do this with the shell option
      LIST(APPEND NODEDUP_XCOMPILER_OPTIONS -Xcompiler)
      LIST(APPEND NODEDUP_XCOMPILER_OPTIONS ${OPT})
    ENDFOREACH()
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_XCOMPILER_OPTIONS}>
    )
  ENDIF()

  IF (MORPHEUS_CXX_STANDARD_FEATURE)
    #GREAT! I can do this the right way
    TARGET_COMPILE_FEATURES(${LIBRARY_NAME} PUBLIC ${MORPHEUS_CXX_STANDARD_FEATURE})
    IF (NOT MORPHEUS_USE_CXX_EXTENSIONS)
      SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES CXX_EXTENSIONS OFF)
    ENDIF()
  ELSE()
    #OH, well, no choice but the wrong way
    TARGET_COMPILE_OPTIONS(${LIBRARY_NAME} PUBLIC ${MORPHEUS_CXX_STANDARD_FLAG})
  ENDIF()
ENDFUNCTION()

FUNCTION(MORPHEUS_LIB_INCLUDE_DIRECTORIES TARGET)
  KOKKOS_LIB_TYPE(${TARGET} INCTYPE)
  FOREACH(DIR ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${TARGET} ${INCTYPE} $<BUILD_INTERFACE:${DIR}>)
  ENDFOREACH()
ENDFUNCTION()