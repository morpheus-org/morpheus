INCLUDE(CMakeParseArguments)
INCLUDE(CTest)

cmake_policy(SET CMP0054 NEW)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

MACRO(GLOBAL_SET VARNAME)
  SET(${VARNAME} ${ARGN} CACHE INTERNAL "")
ENDMACRO()

FUNCTION(GLOBAL_APPEND VARNAME)
  #We make this a function since we are setting variables
  #and want to use scope to avoid overwriting local variables
  SET(TEMP ${${VARNAME}})
  LIST(APPEND TEMP ${ARGN})
  GLOBAL_SET(${VARNAME} ${TEMP})
ENDFUNCTION()

FUNCTION(MOPRHEUS_LIB_TYPE LIB RET)
GET_TARGET_PROPERTY(PROP ${LIB} TYPE)
IF (${PROP} STREQUAL "INTERFACE_LIBRARY")
  SET(${RET} "INTERFACE" PARENT_SCOPE)
ELSE()
  SET(${RET} "PUBLIC" PARENT_SCOPE)
ENDIF()
ENDFUNCTION()

FUNCTION(MORPHEUS_INCLUDE_DIRECTORIES)
  CMAKE_PARSE_ARGUMENTS(
    INC
    "REQUIRED_DURING_INSTALLATION_TESTING"
    ""
    ""
    ${ARGN}
  )
  INCLUDE_DIRECTORIES(${INC_UNPARSED_ARGUMENTS})
ENDFUNCTION()

FUNCTION(MORPHEUS_LIB_TYPE LIB RET)
GET_TARGET_PROPERTY(PROP ${LIB} TYPE)
IF (${PROP} STREQUAL "INTERFACE_LIBRARY")
  SET(${RET} "INTERFACE" PARENT_SCOPE)
ELSE()
  SET(${RET} "PUBLIC" PARENT_SCOPE)
ENDIF()
ENDFUNCTION()