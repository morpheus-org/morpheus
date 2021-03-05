# Library Targets
add_library(morpheus INTERFACE)
add_library(morpheus::morpheus ALIAS morpheus)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
target_include_directories(morpheus INTERFACE 
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include>)

# Find Dependencies
# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
	message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# CUDA
if(MORPHEUS_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	# list(APPEND Morpheus_EXTERNAL_LIBS CUDA::cudart CUDA::cublas)
	target_link_libraries(morpheus INTERFACE CUDA::cudart)	# TODO: Link cublas as math lib
	target_compile_features(morpheus INTERFACE cuda_std_17)
endif()

# Thrust
# TODO: If ROCM backend selected use RocmThrust Instead
if(Morpheus_THRUST)
	morpheus_configure_thrust()
	# TODO: Remove strict dependency on version
	find_package(Thrust ${Morpheus_MIN_VERSION_Thrust} REQUIRED CONFIG)
	message(STATUS "Found Thrust: ${Thrust_DIR} (found version \"${Thrust_VERSION}\") ")
	thrust_create_target(morpheus::Thrust FROM_OPTIONS)
	target_link_libraries(morpheus INTERFACE morpheus::Thrust)
endif()
           
target_compile_features(morpheus INTERFACE cxx_std_17)

# Install Targets
include(GNUInstallDirs)

set(Morpheus_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake" CACHE STRING "install path for MorpheusConfig.cmake")

install(TARGETS morpheus
        EXPORT morpheus-targets)

install(DIRECTORY "${Morpheus_INCLUDE_DIR}"
	TYPE INCLUDE
	FILES_MATCHING
	PATTERN "*.hpp"
	PATTERN "*.inl"
	)

include(CMakePackageConfigHelpers)
configure_package_config_file(${Morpheus_SOURCE_DIR}/cmake/MorpheusConfig.cmake.in
                              ${Morpheus_BINARY_DIR}/cmake/MorpheusConfig.cmake
                              INSTALL_DESTINATION ${Morpheus_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${Morpheus_BINARY_DIR}/cmake/MorpheusConfigVersion.cmake
                                 VERSION ${Morpheus_VERSION}
                                 COMPATIBILITY SameMajorVersion)

install(EXPORT morpheus-targets
		FILE MorpheusTargets.cmake
		NAMESPACE morpheus::
		DESTINATION ${Morpheus_CMAKECONFIG_INSTALL_DIR})

install(FILES ${Morpheus_BINARY_DIR}/cmake/MorpheusConfig.cmake ${Morpheus_BINARY_DIR}/cmake/MorpheusConfigVersion.cmake
		# ${Morpheus_SOURCE_DIR}/cmake/install/FindThrust.cmake	# Remember to install FindThrust.cmake
		DESTINATION ${Morpheus_CMAKECONFIG_INSTALL_DIR})

# Build export targets
set(Morpheus_BUILD_DIR_EXPORT_SETTINGS
    "list(PREPEND CMAKE_MODULE_PATH \"${Morpheus_SOURCE_DIR}/cmake/install/\")")
	 
configure_package_config_file(${Morpheus_SOURCE_DIR}/cmake/MorpheusConfig.cmake.in ${Morpheus_BINARY_DIR}/MorpheusConfig.cmake
                              INSTALL_DESTINATION ${Morpheus_BINARY_DIR})

write_basic_package_version_file(${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake
                                 COMPATIBILITY SameMinorVersion)

export(TARGETS morpheus
		FILE ${Morpheus_BINARY_DIR}/morpheus-targets.cmake
		NAMESPACE morpheus::)

# Packaging
configure_file(${Morpheus_SOURCE_DIR}/cmake/Morpheus.pc.in
              "${Morpheus_BINARY_DIR}/Morpheus.pc"
              @ONLY)

install(FILES "${Morpheus_BINARY_DIR}/Morpheus.pc"
        DESTINATION "${Morpheus_CMAKECONFIG_INSTALL_DIR}/pkgconfig/")