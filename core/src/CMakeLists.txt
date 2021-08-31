morpheus_configure_file(MorpheusCore_config.hpp)

morpheus_include_directories(
  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  ${MORPHEUS_TOP_BUILD_DIR})

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${MORPHEUS_HEADER_INSTALL_DIR}
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

# ADD ELEMENTWISE for DenseVector
if(MORPHEUS_ENABLE_CUDA)
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

  if(MORPHEUS_ENABLE_SERIAL)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Serial/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Serial/*.hpp)
  endif()

  if(MORPHEUS_ENABLE_OPENMP)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/OpenMP/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/OpenMP/*.hpp)
  endif()

  if(MORPHEUS_ENABLE_CUDA)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Cuda/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/Cuda/*.hpp)
  endif()

  if(MORPHEUS_ENABLE_HIP)
    append_glob(MORPHEUS_CORE_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/HIP/*.cpp)
    append_glob(MORPHEUS_CORE_HEADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/${Fmt}/HIP/*.hpp)
  endif()

  if(MORPHEUS_ENABLE_CUDA OR MORPHEUS_ENABLE_HIP)
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
  morpheuscore SOURCES ${MORPHEUS_CORE_SOURCES} HEADERS
  ${MORPHEUS_CORE_HEADERS} ADD_BUILD_OPTIONS # core should be given all the
                                             # necessary compiler/linker flags
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MorpheusCore_config.hpp
        DESTINATION ${MORPHEUS_HEADER_INSTALL_DIR})

morpheus_lib_include_directories(
  morpheuscore ${MORPHEUS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

# Link with Kokkos
target_link_libraries(morpheuscore PUBLIC Kokkos::kokkos)

foreach(HEADER ${MORPHEUS_CORE_HEADERS})
  get_filename_component(dirname ${HEADER} DIRECTORY)
  target_include_directories(morpheuscore PUBLIC $<BUILD_INTERFACE:${dirname}>)
endforeach()
target_include_directories(
  morpheuscore PUBLIC $<INSTALL_INTERFACE:${MORPHEUS_HEADER_INSTALL_DIR}>)

# morpheus_link_tpl(morpheuscore PUBLIC BLAS) morpheus_link_tpl(morpheuscore
# PUBLIC CBLAS) morpheus_link_tpl(morpheuscore PUBLIC MKL)
# morpheus_link_tpl(morpheuscore PUBLIC CUBLAS) morpheus_link_tpl(morpheuscore
# PUBLIC CUSPARSE) morpheus_link_tpl(morpheuscore PUBLIC METIS)
# morpheus_link_tpl(morpheuscore PUBLIC ARMPL)
