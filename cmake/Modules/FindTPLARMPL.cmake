# Both the armpl_mp and armpl libraries define the same public symbol names.
# In order to link against the openmp  armpl symbols, instruct cmake to link against armpl_mp.
# In order to link against the default armpl symbols, instruct cmake to link against armpl.
IF(MORPHEUS_ENABLE_OPENMP)
  SET(ARMPL_LIB armpl_mp)
  message(STATUS "Building ARMPL in OpenMP Mode")
ELSE()
  SET(ARMPL_LIB armpl)
  message(STATUS "Building ARMPL in Serial Mode")
ENDIF()

IF (ARMPL_LIBRARY_DIRS AND ARMPL_LIBRARIES)
  morpheus_find_imported(ARMPL INTERFACE LIBRARIES ${ARMPL_LIBRARIES} LIBRARY_PATHS ${ARMPL_LIBRARY_DIRS})
ELSEIF (ARMPL_LIBRARIES)
  morpheus_find_imported(ARMPL INTERFACE LIBRARIES ${ARMPL_LIBRARIES})
ELSEIF (ARMPL_LIBRARY_DIRS)
  morpheus_find_imported(ARMPL INTERFACE LIBRARIES amath ${ARMPL_LIB} LIBRARY_PATHS ${ARMPL_LIBRARY_DIRS})
ELSEIF (DEFINED ENV{ARMPL_DIR})
  SET(ARMPL_BUILD $ENV{ARMPL_BUILD})
  SET(ARMPL_ROOT $ENV{ARMPL_DIR})
  morpheus_find_imported(ARMPL INTERFACE
    LIBRARIES
      amath
      ${ARMPL_LIB}
    LIBRARY_PATHS
      ${ARMPL_ROOT}/lib
    HEADERS
      armpl.h
    HEADER_PATHS
      ${ARMPL_ROOT}/include
  )
ELSE()
  FIND_PACKAGE(ARMPL REQUIRED)
  morpheus_create_imported_tpl(ARMPL INTERFACE LINK_LIBRARIES ${ARMPL_LIBRARIES})
ENDIF()

# TRY_COMPILE(MORPHEUS_TRY_COMPILE_ARMPL
#   ${MORPHEUS_TOP_BUILD_DIR}/tpl_tests
#   ${MORPHEUS_TOP_SOURCE_DIR}/cmake/compile_tests/armpl.cpp
#   LINK_LIBRARIES -l${ARMPL_LIB} -lgfortran -lamath -lm
#   OUTPUT_VARIABLE MORPHEUS_TRY_COMPILE_ARMPL_OUT)
# IF(NOT MORPHEUS_TRY_COMPILE_ARMPL)
#   MESSAGE(FATAL_ERROR "MORPHEUS_TRY_COMPILE_ARMPL_ARMPL_OUT=${MORPHEUS_TRY_COMPILE_ARMPL_OUT}")
# ELSE()
  # Morpheus::ARMPL is an alias to the ARMPL target.
  # Let's add in the libgfortran and libm dependencies for users here.
  GET_TARGET_PROPERTY(ARMPL_INTERFACE_LINK_LIBRARIES Morpheus::ARMPL INTERFACE_LINK_LIBRARIES)
  SET(ARMPL_INTERFACE_LINK_LIBRARIES "${ARMPL_INTERFACE_LINK_LIBRARIES};-lgfortran;-lm")
  SET_TARGET_PROPERTIES(ARMPL PROPERTIES INTERFACE_LINK_LIBRARIES "${ARMPL_INTERFACE_LINK_LIBRARIES}")
# ENDIF()