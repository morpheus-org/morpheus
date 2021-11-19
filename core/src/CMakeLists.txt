morpheus_configure_file(MorpheusCore_config.hpp)

morpheus_include_directories(
  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  ${MORPHEUS_TOP_BUILD_DIR})

set(MORPHEUS_CORE_HEADER_INSTALL_DIR ${MORPHEUS_HEADER_INSTALL_DIR}/core)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${MORPHEUS_CORE_HEADER_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

set(MORPHEUS_CORE_SOURCES)
append_glob(MORPHEUS_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.cpp)

set(MORPHEUS_CORE_HEADERS)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/fwd/algorithms/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/fwd/containers/*.hpp)
append_glob(MORPHEUS_CORE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/fwd/*.hpp)

foreach(Fmt DenseVector;DenseMatrix)
  append_glob(MORPHEUS_CORE_SOURCES
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/*.hpp)
endforeach()

if(Morpheus_ENABLE_CUDA)
  append_glob(MORPHEUS_CORE_SOURCES
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/DenseVector/Cuda/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/DenseVector/Cuda/*.hpp)
  append_glob(MORPHEUS_CORE_SOURCES
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/DenseVector/Kernels/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/DenseVector/Kernels/*.hpp)
endif()

foreach(Fmt Coo;Csr;Dia)
  append_glob(MORPHEUS_CORE_SOURCES
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/*.cpp)
  append_glob(MORPHEUS_CORE_HEADERS
              ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/*.hpp)

  if(Morpheus_ENABLE_SERIAL)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Serial/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Serial/*.hpp)
  endif()

  if(Morpheus_ENABLE_OPENMP)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/OpenMP/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/OpenMP/*.hpp)
  endif()

  if(Morpheus_ENABLE_CUDA)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Cuda/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Cuda/*.hpp)
  endif()

  if(Morpheus_ENABLE_HIP)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/HIP/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/HIP/*.hpp)
  endif()

  if(Morpheus_ENABLE_CUDA OR MORPHEUS_ENABLE_HIP)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Kernels/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Kernels/*.hpp)
  endif()
endforeach()

append_glob(MORPHEUS_CORE_HEADERS
            ${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_config.hpp)

foreach(HEADER ${MORPHEUS_CORE_HEADERS})
  get_filename_component(dirname ${HEADER} DIRECTORY)
  morpheus_include_directories(${dirname})
endforeach()

morpheus_add_library(
  morpheus-core SOURCES ${MORPHEUS_CORE_SOURCES} HEADERS
  ${MORPHEUS_CORE_HEADERS} ADD_BUILD_OPTIONS # core should be given all the
                                             # necessary compiler/linker flags
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_config.hpp
        DESTINATION ${MORPHEUS_CORE_HEADER_INSTALL_DIR})

morpheus_lib_include_directories(
  morpheus-core ${MORPHEUS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

# Link with Kokkos
target_link_libraries(morpheus-core PUBLIC Kokkos::kokkos)

foreach(HEADER ${MORPHEUS_CORE_HEADERS})
  get_filename_component(dirname ${HEADER} DIRECTORY)
  target_include_directories(morpheus-core PUBLIC $<BUILD_INTERFACE:${dirname}>)
endforeach()
target_include_directories(
  morpheus-core PUBLIC $<INSTALL_INTERFACE:${MORPHEUS_CORE_HEADER_INSTALL_DIR}>)
