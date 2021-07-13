macro(GLOBAL_SET VARNAME)
  set(${VARNAME}
      ${ARGN}
      CACHE INTERNAL "")
endmacro()

function(GLOBAL_APPEND VARNAME)
  # We make this a function since we are setting variables and want to use scope
  # to avoid overwriting local variables
  set(TEMP ${${VARNAME}})
  list(APPEND TEMP ${ARGN})
  global_set(${VARNAME} ${TEMP})
endfunction()

macro(APPEND_GLOB VAR)
  file(GLOB LOCAL_TMP_VAR ${ARGN})
  list(APPEND ${VAR} ${LOCAL_TMP_VAR})
endmacro()

macro(MORPHEUS_PACKAGE_DECL)
  set(PACKAGE_NAME Morpheus)
  set(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  set(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endmacro(MORPHEUS_PACKAGE_DECL)
# ~~~
# macro(MORPHEUS_PACKAGE_POSTPROCESS)
#   include(CMakePackageConfigHelpers)
#   configure_package_config_file(
#     cmake/MorpheusConfig.cmake.in "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)
#   write_basic_package_version_file(
#     "${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
#     VERSION
#       "${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR}.${Morpheus_VERSION_PATCH}"
#     COMPATIBILITY SameMajorVersion)
#   install(FILES "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
#                 "${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
#           DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)
#   install(
#     EXPORT MorpheusTargets
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus
#     NAMESPACE Morpheus::)
# endmacro(MORPHEUS_PACKAGE_POSTPROCESS)
# ~~~

macro(MORPHEUS_PROCESS_SUBPACKAGES)
  add_subdirectory(core)
  add_subdirectory(examples)
endmacro()

macro(MORPHEUS_SUBPACKAGE NAME)
  set(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
  set(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  set(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  # ADD_INTERFACE_LIBRARY(PACKAGE_${PACKAGE_NAME})
  # GLOBAL_SET(${PACKAGE_NAME}_LIBS "")
endmacro()

macro(MORPHEUS_ADD_TEST_DIRECTORIES)
  if(MORPHEUS_ENABLE_TESTS)
    foreach(TEST_DIR ${ARGN})
      add_subdirectory(${TEST_DIR})
    endforeach()
  endif()
endmacro()

macro(MORPHEUS_ADD_EXAMPLE_DIRECTORIES)
  if(MORPHEUS_ENABLE_EXAMPLES)
    foreach(EXAMPLE_DIR ${ARGN})
      add_subdirectory(${EXAMPLE_DIR})
    endforeach()
  endif()
endmacro()

function(morpheus_add_option SUFFIX DEFAULT TYPE DOCSTRING)
  cmake_parse_arguments(
    OPT "" "" "VALID_ENTRIES" # if this is a list variable, the valid values in
                              # the list
    ${ARGN})

  set(CAMEL_NAME Morpheus_${SUFFIX})
  string(TOUPPER ${CAMEL_NAME} UC_NAME)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  set(${CAMEL_NAME}
      ${DEFAULT}
      CACHE ${TYPE} ${DOCSTRING})

  foreach(opt ${MORPHEUS_GIVEN_VARIABLES})
    string(TOUPPER ${opt} OPT_UC)
    if("${OPT_UC}" STREQUAL "${UC_NAME}")
      if(NOT "${opt}" STREQUAL "${CAMEL_NAME}")
        message(
          FATAL_ERROR
            "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies."
        )
      endif()
    endif()
  endforeach()

  # okay, great, we passed the validation test - use the default
  if(DEFINED ${CAMEL_NAME})
    if(OPT_VALID_ENTRIES)
      string(TOUPPER "${OPT_VALID_ENTRIES}" OPT_VALID_ENTRIES_UC)
      foreach(entry ${${CAMEL_NAME}})
        string(TOUPPER ${entry} ENTRY_UC)
        if(NOT ${ENTRY_UC} IN_LIST OPT_VALID_ENTRIES_UC)
          message(
            FATAL_ERROR
              "Given entry ${entry} in list for option ${SUFFIX}. "
              "Valid case-insensitive values are any of ${OPT_VALID_ENTRIES}")
        endif()
      endforeach()
      string(TOUPPER "${${CAMEL_NAME}}" GIVEN_ENTRIES_UC)
      set(${UC_NAME}
          ${GIVEN_ENTRIES_UC}
          PARENT_SCOPE)
    else()
      set(${UC_NAME}
          ${${CAMEL_NAME}}
          PARENT_SCOPE)
    endif()
  else()
    set(${UC_NAME}
        ${DEFAULT}
        PARENT_SCOPE)
  endif()

endfunction()
