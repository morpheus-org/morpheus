morpheus_configure_file(Morpheus_config.h)

morpheus_include_directories(
  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  ${MORPHEUS_TOP_BUILD_DIR})

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${MORPHEUS_HEADER_DIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

# I don't want the complexity of sometimes interface/sometimes not Just
# minimally force an empty dummy file
set(MORPHEUS_CORE_SRCS dummy.cpp)
append_glob(MORPHEUS_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.cpp)
append_glob(MORPHEUS_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(MORPHEUS_CORE_HEADERS)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/fwd/*.hpp)

if(MORPEHUS_ENABLE_CUDA)
  append_glob(MORPHEUS_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/impl/Cuda/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/impl/Cuda/*.hpp)
endif()

if(MORPEHUS_ENABLE_OPENMP)
  append_glob(MORPHEUS_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/impl/OpenMP/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/OpenMP/*.hpp)
endif()

if(MORPEHUS_ENABLE_SERIAL)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/Serial/*.hpp)
endif()

append_glob(MORPHEUS_CORE_HEADERS
            ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h)

morpheus_add_library(
  morpheuscore SOURCES ${MORPHEUS_CORE_SRCS} HEADERS ${MORPHEUS_CORE_HEADERS}
  ADD_BUILD_OPTIONS # core should be given all the necessary compiler/linker
                    # flags
)

morpheus_lib_include_directories(
  morpheuscore ${MORPHEUS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
# Link with Kokkos
target_link_libraries(morpheuscore PUBLIC Kokkos::kokkos)
foreach(DIR ${MORPHEUS_CORE_HEADERS})
  target_include_directories(morpheuscore PUBLIC $<BUILD_INTERFACE:${DIR}>)
endforeach()
target_include_directories(morpheuscore
                           PUBLIC $<INSTALL_INTERFACE:${MORPHEUS_HEADER_DIR}>)
