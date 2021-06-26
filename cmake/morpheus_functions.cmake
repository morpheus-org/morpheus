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