INCLUDE(CMakePackageConfigHelpers)
IF (NOT Morphues_INSTALL_TESTING)
  INCLUDE(GNUInstallDirs)

  #Set all the variables needed for MorpheusConfig.cmake
  GET_PROPERTY(MORPHEUS_PROP_LIBS GLOBAL PROPERTY MORPHEUS_LIBRARIES_NAMES)
  SET(MORPHEUS_LIBRARIES ${MORPHEUS_PROP_LIBS})

  INCLUDE(CMakePackageConfigHelpers)
  CONFIGURE_PACKAGE_CONFIG_FILE(
    cmake/MorpheusConfig.cmake.in
    "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)

  CONFIGURE_PACKAGE_CONFIG_FILE(
	  cmake/MorpheusConfigCommon.cmake.in
	  "${Morpheus_BINARY_DIR}/MorpheusConfigCommon.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)

  WRITE_BASIC_PACKAGE_VERSION_FILE("${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
      VERSION "${Morpheus_VERSION}"
      COMPATIBILITY SameMajorVersion)

  # Install the MorpheusConfig*.cmake files
  install(FILES
    "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
    "${Morpheus_BINARY_DIR}/MorpheusConfigCommon.cmake"
    "${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)
  install(EXPORT MorpheusTargets NAMESPACE Morpheus:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)
ELSE()
  CONFIGURE_FILE(cmake/MorpheusConfigCommon.cmake.in ${Morpheus_BINARY_DIR}/MorpheusConfigCommon.cmake @ONLY)
  file(READ ${Morpheus_BINARY_DIR}/MorpheusConfigCommon.cmake MORPHEUS_CONFIG_COMMON)
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MorpheusConfig_install.cmake" "${MORPHEUS_CONFIG_COMMON}")
  CONFIGURE_FILE(cmake/MorpheusTrilinosConfig.cmake.in ${Morpheus_BINARY_DIR}/MorpheusTrilinosConfig.cmake @ONLY)
  file(READ ${Morpheus_BINARY_DIR}/MorpheusTrilinosConfig.cmake MORPHEUS_TRILINOS_CONFIG)
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MorpheusConfig_install.cmake" "${MORPHEUS_TRILINOS_CONFIG}")

  WRITE_BASIC_PACKAGE_VERSION_FILE("${CMAKE_CURRENT_BINARY_DIR}/MorpheusConfigVersion.cmake"
      VERSION "${Morpheus_VERSION}"
      COMPATIBILITY SameMajorVersion)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MorpheusConfigVersion.cmake
      DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/${PACKAGE_NAME}")
ENDIF()

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_config.hpp DESTINATION ${Morpheus_HEADER_DIR})