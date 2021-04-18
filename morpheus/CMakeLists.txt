# Library Targets
add_library(morpheus INTERFACE)
add_library(Morpheus::morpheus ALIAS morpheus)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
target_include_directories(morpheus INTERFACE 
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include>)

find_package(Kokkos REQUIRED)
target_link_libraries(morpheus INTERFACE Kokkos::kokkos)

target_compile_features(morpheus INTERFACE cxx_std_17)
set(gnu_cxx "$<COMPILE_LANG_AND_ID:CXX,GNU>")
set(gnu_cxx_flags 	-Wall -Wshadow -pedantic
					-Werror -Wsign-compare -Wtype-limits
					-Wignored-qualifiers -Wempty-body
					-Wclobbered -Wuninitialized)
set(clang_cxx "$<COMPILE_LANG_AND_ID:CXX,RMClang,AppleClang,Clang>")
set(clang_cxx_flags -Wall -Wshadow -pedantic
					-Werror -Wsign-compare -Wtype-limits
					-Wignored-qualifiers -Wempty-body
					-Wclobbered -Wuninitialized)
target_compile_options(morpheus INTERFACE
	"$<${gnu_cxx}:$<BUILD_INTERFACE:${gnu_cxx_flags}>>"
	"$<${clang_cxx}:$<BUILD_INTERFACE:${clang_cxx_flags}>>")
# Install Targets
include(GNUInstallDirs)

set(Morpheus_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus" CACHE STRING "install path for MorpheusConfig.cmake")

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
		NAMESPACE Morpheus::
		DESTINATION ${Morpheus_CMAKECONFIG_INSTALL_DIR})

install(FILES ${Morpheus_BINARY_DIR}/cmake/MorpheusConfig.cmake ${Morpheus_BINARY_DIR}/cmake/MorpheusConfigVersion.cmake
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
		NAMESPACE Morpheus::)

# Packaging
configure_file(${Morpheus_SOURCE_DIR}/cmake/Morpheus.pc.in
              "${Morpheus_BINARY_DIR}/Morpheus.pc"
              @ONLY)

install(FILES "${Morpheus_BINARY_DIR}/Morpheus.pc"
        DESTINATION "${Morpheus_CMAKECONFIG_INSTALL_DIR}/pkgconfig/")