cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3")

set(Thrust_DIR "/Users/cstyl/Desktop/Projects/thrust/thrust/cmake/")
set(Cusp_DIR "/Users/cstyl/Desktop/Projects/cusplibrary-0.5.1/")
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)

include_directories(${Cusp_DIR})
include_directories(/Users/cstyl/Desktop/Projects/morpheus)

set(BOOST_ROOT "/Users/cstyl/Desktop/Projects/boost_1_73_0")

find_package(Boost 1.73.0)
if(Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_executable(static_spmv_coo static_spmv_coo.cpp)
add_executable(dynamic_spmv_coo dynamic_spmv_coo.cpp)
add_executable(cusp_spmv_coo cusp_spmv_coo.cpp)

if(Boost_FOUND)
    target_link_libraries(static_spmv_coo ${Boost_LIBRARIES})
    target_link_libraries(dynamic_spmv_coo ${Boost_LIBRARIES})
    target_link_libraries(cusp_spmv_coo ${Boost_LIBRARIES})
endif()

target_link_libraries(static_spmv_coo ThrustOMP)
target_link_libraries(dynamic_spmv_coo ThrustOMP)
target_link_libraries(cusp_spmv_coo ThrustOMP)