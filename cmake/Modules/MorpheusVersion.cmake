# Generate version.hpp from the version found in CMakeLists.txt
function(write_version)
  message(STATUS "Morpheus Version: ${Morpheus_VERSION}")
  configure_file(${Morpheus_SOURCE_DIR}/cmake/version.hpp.in
                 ${Morpheus_SOURCE_DIR}/morpheus/version.hpp @ONLY)
endfunction()
