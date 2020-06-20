cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3")

# Archer directories
set(Local_DIR "/work/e609/e609/cstyl")

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

add_executable(morpheus main.cpp)

if(Boost_FOUND)
    target_link_libraries(morpheus ${Boost_LIBRARIES})
endif()

target_link_libraries(morpheus ThrustOMP)
