# @FUNCTION: morpheus_option
#
# Validate options are given with correct case and define an internal
# upper-case version for use within
FUNCTION(morpheus_option CAMEL_SUFFIX DEFAULT TYPE DOCSTRING)
  SET(CAMEL_NAME Morpheus_${CAMEL_SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  LIST(APPEND MORPHEUS_OPTION_KEYS ${CAMEL_SUFFIX})
  SET(MORPHEUS_OPTION_KEYS ${MORPHEUS_OPTION_KEYS} PARENT_SCOPE)
  LIST(APPEND MORPHEUS_OPTION_VALUES "${DOCSTRING}")
  SET(MORPHEUS_OPTION_VALUES ${MORPHEUS_OPTION_VALUES} PARENT_SCOPE)
  LIST(APPEND MORPHEUS_OPTION_TYPES ${TYPE})
  SET(MORPHEUS_OPTION_TYPES ${MORPHEUS_OPTION_TYPES} PARENT_SCOPE)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  SET(${CAMEL_NAME} ${DEFAULT} CACHE ${TYPE} ${DOCSTRING})

  #I don't love doing it this way because it's N^2 in number options, but cest la vie
  FOREACH(opt ${MORPHEUS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      IF (NOT "${opt}" STREQUAL "${CAMEL_NAME}")
          MESSAGE(FATAL_ERROR "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies.")
      ENDIF()
    ENDIF()
  ENDFOREACH()

  #okay, great, we passed the validation test - use the default
  IF (DEFINED ${CAMEL_NAME})
    SET(${UC_NAME} ${${CAMEL_NAME}} PARENT_SCOPE)
  ELSE()
    SET(${UC_NAME} ${DEFAULT} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

FUNCTION(morpheus_set_option CAMEL_SUFFIX VALUE)
  LIST(FIND MORPHEUS_OPTION_KEYS ${CAMEL_SUFFIX} OPTION_INDEX)
  IF(OPTION_INDEX EQUAL -1)
    MESSAGE(FATAL_ERROR "Couldn't set value for Morpheus_${CAMEL_SUFFIX}")
  ENDIF()
  SET(CAMEL_NAME Morpheus_${CAMEL_SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  LIST(GET MORPHEUS_OPTION_VALUES ${OPTION_INDEX} DOCSTRING)
  LIST(GET MORPHEUS_OPTION_TYPES ${OPTION_INDEX} TYPE)
  SET(${CAMEL_NAME} ${VALUE} CACHE ${TYPE} ${DOCSTRING} FORCE)
  MESSAGE(STATUS "Setting ${CAMEL_NAME}=${VALUE}")
  SET(${UC_NAME} ${VALUE} PARENT_SCOPE)
ENDFUNCTION()

# @FUNCTION: morpheus_deprecated_list
#
# Function that checks if a deprecated list option like Morpheus_ARCH was given.
# This prints an error and prevents configure from completing.
# It attempts to print a helpful message about updating the options for the new CMake.
# Morpheus_${SUFFIX} is the name of the option (like Morpheus_ARCH) being checked.
# Morpheus_${PREFIX}_X is the name of new option to be defined from a list X,Y,Z,...
FUNCTION(morpheus_deprecated_list SUFFIX PREFIX)
  SET(CAMEL_NAME Morpheus_${SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  #I don't love doing it this way but better to be safe
  FOREACH(opt ${MORPHEUS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      STRING(REPLACE "," ";" optlist "${${opt}}")
      SET(ERROR_MSG "Given deprecated option list ${opt}. This must now be given as separate -D options, which assuming you spelled options correctly would be:")
      FOREACH(entry ${optlist})
        STRING(TOUPPER ${entry} ENTRY_UC)
        STRING(APPEND ERROR_MSG "\n  -DMorpheus_${PREFIX}_${ENTRY_UC}=ON")
      ENDFOREACH()
      STRING(APPEND ERROR_MSG "\nRemove CMakeCache.txt and re-run. For a list of valid options, refer to BUILD.md or even look at CMakeCache.txt (before deleting it).")
      MESSAGE(SEND_ERROR ${ERROR_MSG})
    ENDIF()
  ENDFOREACH()
ENDFUNCTION()


## MORPHEUS_CONFIG_HEADER - parse the data list which is a list of backend names
##                        and create output config header file...used for
##                        creating dynamic include files based on enabled backends
##
##                        SRC_FILE is input file
##                        TARGET_FILE output file
##                        HEADER_GUARD TEXT used with include header guard
##                        HEADER_PREFIX prefix used with include (i.e. fwd, decl, setup)
##                        DATA_LIST list of backends to include in generated file
FUNCTION(MORPHEUS_CONFIG_HEADER SRC_FILE TARGET_FILE HEADER_GUARD HEADER_PREFIX DATA_LIST)
   SET(HEADER_GUARD_TAG "${HEADER_GUARD}_HPP")
   CONFIGURE_FILE(cmake/${SRC_FILE} ${PROJECT_BINARY_DIR}/temp/${TARGET_FILE}.work COPYONLY)
   FOREACH( BACKEND_NAME ${DATA_LIST} )
   SET(INCLUDE_NEXT_FILE "#include <${HEADER_PREFIX}_${BACKEND_NAME}.hpp>
\@INCLUDE_NEXT_FILE\@")
   CONFIGURE_FILE(${PROJECT_BINARY_DIR}/temp/${TARGET_FILE}.work ${PROJECT_BINARY_DIR}/temp/${TARGET_FILE}.work @ONLY)
   ENDFOREACH()
   SET(INCLUDE_NEXT_FILE "" )
   CONFIGURE_FILE(${PROJECT_BINARY_DIR}/temp/${TARGET_FILE}.work ${TARGET_FILE} @ONLY)
ENDFUNCTION()

#
# @MACRO: MORPHEUS_LINK_TPL()
#
# Function that checks if a third-party library (TPL) has been enabled and
# calls target_link_libraries on the given target
#
# Usage::
#
#   MORPHEUS_LINK_TPL(
#     <TARGET>
#     PUBLIC
#     PRIVATE
#     INTERFACE
#     IMPORTED_NAME  <name>
#     <TPL_NAME>
#
#   Checks if Morpheus_ENABLE_<TPL_NAME>=ON and if so links the library
#
#   ``PUBLIC/PRIVATE/INTERFACE``
#
#     Specifies the linkage mode. One of these arguments should be given.
#     This will then invoke target_link_libraries(<TARGET> PUBLIC/PRIVATE/INTERFACE <TPL_NAME>)
#
#   ``IMPORTED_NAME <name>``
#
#     If specified, this gives the exact name of the target to link against
#     target_link_libraries(<TARGET> <IMPORTED_NAME>)
#
FUNCTION(morpheus_link_tpl TARGET)
  CMAKE_PARSE_ARGUMENTS(TPL
   "PUBLIC;PRIVATE;INTERFACE"
   "IMPORTED_NAME"
   ""
   ${ARGN})
  #the name of the TPL
  SET(TPL ${TPL_UNPARSED_ARGUMENTS})
  
  IF (NOT TPL_IMPORTED_NAME)
    SET(TPL_IMPORTED_NAME Morpheus::${TPL})
  ENDIF()
  IF (MORPHEUS_ENABLE_${TPL})
    IF (TPL_PUBLIC)
      TARGET_LINK_LIBRARIES(${TARGET} PUBLIC ${TPL_IMPORTED_NAME})
    ELSEIF (TPL_PRIVATE)
      TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ${TPL_IMPORTED_NAME})
    ELSEIF (TPL_INTERFACE)
      TARGET_LINK_LIBRARIES(${TARGET} INTERFACE ${TPL_IMPORTED_NAME})
    ELSE()
      TARGET_LINK_LIBRARIES(${TARGET} ${TPL_IMPORTED_NAME})
    ENDIF()
  ENDIF()
ENDFUNCTION()

FUNCTION(COMPILER_SPECIFIC_OPTIONS_HELPER)
  SET(COMPILERS NVIDIA PGI DEFAULT Cray Intel Clang AppleClang IntelLLVM GNU HIPCC)
  CMAKE_PARSE_ARGUMENTS(
    PARSE
    "LINK_OPTIONS;COMPILE_OPTIONS;COMPILE_DEFINITIONS;LINK_LIBRARIES"
    "COMPILER_ID"
    "${COMPILERS}"
    ${ARGN})
  IF(PARSE_UNPARSED_ARGUMENTS)
    MESSAGE(SEND_ERROR "'${PARSE_UNPARSED_ARGUMENTS}' argument(s) not recognized when providing compiler specific options")
  ENDIF()

  IF(PARSE_COMPILER_ID)
    SET(COMPILER ${${PARSE_COMPILER_ID}})
  ELSE()
    SET(COMPILER ${MORPHEUS_CXX_COMPILER_ID})
  ENDIF()

  SET(COMPILER_SPECIFIC_FLAGS_TMP)
  FOREACH(COMP ${COMPILERS})
    IF (COMPILER STREQUAL "${COMP}")
      IF (PARSE_${COMPILER})
        IF (NOT "${PARSE_${COMPILER}}" STREQUAL "NO-VALUE-SPECIFIED")
           SET(COMPILER_SPECIFIC_FLAGS_TMP ${PARSE_${COMPILER}})
        ENDIF()
      ELSEIF(PARSE_DEFAULT)
        SET(COMPILER_SPECIFIC_FLAGS_TMP ${PARSE_DEFAULT})
      ENDIF()
    ENDIF()
  ENDFOREACH()

  IF (PARSE_COMPILE_OPTIONS)
    # The funky logic here is for future handling of argument deduplication
    # If we naively pass multiple -Xcompiler flags to target_compile_options
    # -Xcompiler will get deduplicated and break the build
    IF ("-Xcompiler" IN_LIST COMPILER_SPECIFIC_FLAGS_TMP)
      LIST(REMOVE_ITEM COMPILER_SPECIFIC_FLAGS_TMP "-Xcompiler")
      GLOBAL_APPEND(MORPHEUS_XCOMPILER_OPTIONS ${COMPILER_SPECIFIC_FLAGS_TMP})
    ELSE()
      GLOBAL_APPEND(MORPHEUS_COMPILE_OPTIONS ${COMPILER_SPECIFIC_FLAGS_TMP})
    ENDIF()
  ENDIF()

  IF (PARSE_LINK_OPTIONS)
    GLOBAL_APPEND(MORPHEUS_LINK_OPTIONS ${COMPILER_SPECIFIC_FLAGS_TMP})
  ENDIF()

  IF (PARSE_COMPILE_DEFINITIONS)
    GLOBAL_APPEND(MORPHEUS_COMPILE_DEFINITIONS ${COMPILER_SPECIFIC_FLAGS_TMP})
  ENDIF()

  IF (PARSE_LINK_LIBRARIES)
    GLOBAL_APPEND(MORPHEUS_LINK_LIBRARIES ${COMPILER_SPECIFIC_FLAGS_TMP})
  ENDIF()
ENDFUNCTION(COMPILER_SPECIFIC_OPTIONS_HELPER)

FUNCTION(COMPILER_SPECIFIC_FLAGS)
  COMPILER_SPECIFIC_OPTIONS_HELPER(${ARGN} COMPILE_OPTIONS LINK_OPTIONS)
ENDFUNCTION(COMPILER_SPECIFIC_FLAGS)

FUNCTION(COMPILER_SPECIFIC_OPTIONS)
  COMPILER_SPECIFIC_OPTIONS_HELPER(${ARGN} COMPILE_OPTIONS)
ENDFUNCTION(COMPILER_SPECIFIC_OPTIONS)

FUNCTION(COMPILER_SPECIFIC_LINK_OPTIONS)
  COMPILER_SPECIFIC_OPTIONS_HELPER(${ARGN} LINK_OPTIONS)
ENDFUNCTION(COMPILER_SPECIFIC_LINK_OPTIONS)

FUNCTION(COMPILER_SPECIFIC_DEFS)
  COMPILER_SPECIFIC_OPTIONS_HELPER(${ARGN} COMPILE_DEFINITIONS)
ENDFUNCTION(COMPILER_SPECIFIC_DEFS)

FUNCTION(COMPILER_SPECIFIC_LIBS)
  COMPILER_SPECIFIC_OPTIONS_HELPER(${ARGN} LINK_LIBRARIES)
ENDFUNCTION(COMPILER_SPECIFIC_LIBS)

# Given a list of the form
#  key1;value1;key2;value2,...
# Create a list of all keys in a variable named ${KEY_LIST_NAME}
# and set the value for each key in a variable ${VAR_PREFIX}key1,...
# morpheus_key_value_map(ARCH ALL_ARCHES key1;value1;key2;value2)
# would produce a list variable ALL_ARCHES=key1;key2
# and individual variables ARCHkey1=value1 and ARCHkey2=value2
MACRO(MORPHEUS_KEY_VALUE_MAP VAR_PREFIX KEY_LIST_NAME)
  SET(PARSE_KEY ON)
  SET(${KEY_LIST_NAME})
  FOREACH(ENTRY ${ARGN})
    IF(PARSE_KEY)
      SET(CURRENT_KEY ${ENTRY})
      SET(PARSE_KEY OFF)
      LIST(APPEND ${KEY_LIST_NAME} ${CURRENT_KEY})
    ELSE()
      SET(${VAR_PREFIX}${CURRENT_KEY} ${ENTRY})
      SET(PARSE_KEY ON)
    ENDIF()
  ENDFOREACH()
ENDMACRO()

FUNCTION(MORPHEUS_CHECK_DEPRECATED_OPTIONS)
  MORPHEUS_KEY_VALUE_MAP(DEPRECATED_MSG_ DEPRECATED_LIST ${ARGN})
  FOREACH(OPTION_SUFFIX ${DEPRECATED_LIST})
    SET(OPTION_NAME Morpheus_${OPTION_SUFFIX})
    SET(OPTION_MESSAGE ${DEPRECATED_MSG_${OPTION_SUFFIX}})
    IF(DEFINED ${OPTION_NAME}) # This variable has been given by the user as on or off
      MESSAGE(SEND_ERROR "Removed option ${OPTION_NAME} has been given with value ${${OPTION_NAME}}. ${OPT_MESSAGE}")
    ENDIF()
  ENDFOREACH()
ENDFUNCTION()

FUNCTION(morpheus_append_config_line LINE)
  GLOBAL_APPEND(MORPHEUS_TPL_EXPORTS "${LINE}")
ENDFUNCTION()

#
# @MACRO: MORPHEUS_IMPORT_TPL()
#
# Function that checks if a third-party library (TPL) has been enabled and calls `find_package`
# to create an imported target encapsulating all the flags and libraries
# needed to use the TPL
#
# Usage::
#
#   MORPHEUS_IMPORT_TPL(
#     <NAME>
#     NO_EXPORT
#     INTERFACE
#
#   ``NO_EXPORT``
#
#     If specified, this TPL will not be added to MorpheusConfig.cmake as an export
#
#   ``INTERFACE``
#
#     If specified, this TPL will build an INTERFACE library rather than an
#     IMPORTED target
MACRO(morpheus_import_tpl NAME)
  CMAKE_PARSE_ARGUMENTS(TPL
   "NO_EXPORT;INTERFACE"
   ""
   ""
   ${ARGN})
  IF (TPL_INTERFACE)
    SET(TPL_IMPORTED_NAME ${NAME})
  ELSE()
    SET(TPL_IMPORTED_NAME Morpheus::${NAME})
  ENDIF()

  # Even though this policy gets set in the top-level CMakeLists.txt,
  # I have still been getting errors about ROOT variables being ignored
  # I'm not sure if this is a scope issue - but make sure
  # the policy is set before we do any find_package calls
  CMAKE_POLICY(SET CMP0074 NEW)

  IF (MORPHEUS_ENABLE_${NAME})
    #Tack on a TPL here to make sure we avoid using anyone else's find
    FIND_PACKAGE(TPL${NAME} REQUIRED MODULE)
    IF(NOT TARGET ${TPL_IMPORTED_NAME})
      MESSAGE(FATAL_ERROR "Find module succeeded for ${NAME}, but did not produce valid target ${TPL_IMPORTED_NAME}")
    ENDIF()
    IF(NOT TPL_NO_EXPORT)
      MORPHEUS_EXPORT_IMPORTED_TPL(${TPL_IMPORTED_NAME})
    ENDIF()
    LIST(APPEND MORPHEUS_ENABLED_TPLS ${NAME})
  ENDIF()
ENDMACRO(kokkos_import_tpl)

MACRO(morpheus_export_imported_tpl NAME)
    #make sure this also gets "exported" in the config file
    MORPHEUS_APPEND_CONFIG_LINE("IF(NOT TARGET ${NAME})")

    GET_TARGET_PROPERTY(LIB_TYPE ${NAME} TYPE)
    IF (${LIB_TYPE} STREQUAL "INTERFACE_LIBRARY")
      MORPHEUS_APPEND_CONFIG_LINE("ADD_LIBRARY(${NAME} INTERFACE IMPORTED)")
      MORPHEUS_APPEND_CONFIG_LINE("SET_TARGET_PROPERTIES(${NAME} PROPERTIES")
    ELSE()
      MORPHEUS_APPEND_CONFIG_LINE("ADD_LIBRARY(${NAME} UNKNOWN IMPORTED)")
      MORPHEUS_APPEND_CONFIG_LINE("SET_TARGET_PROPERTIES(${NAME} PROPERTIES")
      GET_TARGET_PROPERTY(TPL_LIBRARY ${NAME} IMPORTED_LOCATION)
      IF(TPL_LIBRARY)
        MORPHEUS_APPEND_CONFIG_LINE("IMPORTED_LOCATION \"${TPL_LIBRARY}\"")
      ENDIF()
    ENDIF()

    GET_TARGET_PROPERTY(TPL_INCLUDES ${NAME} INTERFACE_INCLUDE_DIRECTORIES)
    IF(TPL_INCLUDES)
      MORPHEUS_APPEND_CONFIG_LINE("INTERFACE_INCLUDE_DIRECTORIES \"${TPL_INCLUDES}\"")
    ENDIF()

    GET_TARGET_PROPERTY(TPL_COMPILE_OPTIONS ${NAME} INTERFACE_COMPILE_OPTIONS)
    IF(TPL_COMPILE_OPTIONS)
      MORPHEUS_APPEND_CONFIG_LINE("INTERFACE_COMPILE_OPTIONS ${TPL_COMPILE_OPTIONS}")
    ENDIF()

    SET(TPL_LINK_OPTIONS)
    GET_TARGET_PROPERTY(TPL_LINK_OPTIONS ${NAME} INTERFACE_LINK_OPTIONS)
    IF(TPL_LINK_OPTIONS)
      MORPHEUS_APPEND_CONFIG_LINE("INTERFACE_LINK_OPTIONS ${TPL_LINK_OPTIONS}")
    ENDIF()

    GET_TARGET_PROPERTY(TPL_LINK_LIBRARIES  ${NAME} INTERFACE_LINK_LIBRARIES)
    IF(TPL_LINK_LIBRARIES)
      MORPHEUS_APPEND_CONFIG_LINE("INTERFACE_LINK_LIBRARIES \"${TPL_LINK_LIBRARIES}\"")
    ENDIF()
    MORPHEUS_APPEND_CONFIG_LINE(")")
    MORPHEUS_APPEND_CONFIG_LINE("ENDIF()")
  ENDIF()
ENDMACRO()