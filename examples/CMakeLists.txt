cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3")

# Local directories
set(Local_DIR "/Users/cstyl/Desktop/Projects")
# Archer directories
#set(Local_DIR "/work/e609/e609/cstyl")

set(Thrust_DIR "${Local_DIR}/thrust/thrust/cmake")
set(Cusp_DIR "${Local_DIR}/cusplibrary-0.5.1")
set(Morpheus_DIR "${Local_DIR}/morpheus")
set(BOOST_ROOT "${Local_DIR}/boost_1_73_0")

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)

include_directories(${Cusp_DIR})
include_directories(${Morpheus_DIR})

find_package(Boost 1.73.0)
if(Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_executable(cusp_coo_spmv cusp/coo.cpp)
add_executable(cusp_csr_spmv cusp/csr.cpp)
add_executable(cusp_dia_spmv cusp/dia.cpp)
add_executable(cusp_ell_spmv cusp/ell.cpp)
add_executable(cusp_hyb_spmv cusp/hyb.cpp)
add_executable(static_coo_spmv static/coo.cpp)
add_executable(static_csr_spmv static/csr.cpp)
add_executable(static_dia_spmv static/dia.cpp)
add_executable(static_ell_spmv static/ell.cpp)
add_executable(static_hyb_spmv static/hyb.cpp)
add_executable(dynamic_coo_spmv dynamic/coo.cpp)
add_executable(dynamic_csr_spmv dynamic/csr.cpp)
add_executable(dynamic_dia_spmv dynamic/dia.cpp)
add_executable(dynamic_ell_spmv dynamic/ell.cpp)
add_executable(dynamic_hyb_spmv dynamic/hyb.cpp)


if(Boost_FOUND)
    target_link_libraries(dynamic_coo_spmv ${Boost_LIBRARIES})
    target_link_libraries(dynamic_csr_spmv ${Boost_LIBRARIES})
    target_link_libraries(dynamic_dia_spmv ${Boost_LIBRARIES})
    target_link_libraries(dynamic_ell_spmv ${Boost_LIBRARIES})
    target_link_libraries(dynamic_hyb_spmv ${Boost_LIBRARIES})
endif()

target_link_libraries(cusp_coo_spmv ThrustOMP)
target_link_libraries(cusp_csr_spmv ThrustOMP)
target_link_libraries(cusp_dia_spmv ThrustOMP)
target_link_libraries(cusp_ell_spmv ThrustOMP)
target_link_libraries(cusp_hyb_spmv ThrustOMP)
target_link_libraries(static_coo_spmv ThrustOMP)
target_link_libraries(static_csr_spmv ThrustOMP)
target_link_libraries(static_dia_spmv ThrustOMP)
target_link_libraries(static_ell_spmv ThrustOMP)
target_link_libraries(static_hyb_spmv ThrustOMP)
target_link_libraries(dynamic_coo_spmv ThrustOMP)
target_link_libraries(dynamic_csr_spmv ThrustOMP)
target_link_libraries(dynamic_dia_spmv ThrustOMP)
target_link_libraries(dynamic_ell_spmv ThrustOMP)
target_link_libraries(dynamic_hyb_spmv ThrustOMP)