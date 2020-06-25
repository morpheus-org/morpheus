cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 14)

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
else()
    message(FATAL_ERROR "BOOST not found")
endif()


add_executable(cusp src/cusp.cpp)
target_compile_options(cusp PUBLIC -O3 -Winline)
target_link_libraries(cusp ThrustOMP)

add_executable(dynamic_1 src/dynamic.cpp)
target_compile_definitions(dynamic_1 PUBLIC -DNUM_TYPES=1)
target_compile_options(dynamic_1 PUBLIC -O3 -Winline)
target_link_libraries(dynamic_1 ${Boost_LIBRARIES})
target_link_libraries(dynamic_1 ThrustOMP)

add_executable(dynamic_6 src/dynamic.cpp)
target_compile_definitions(dynamic_6 PUBLIC -DNUM_TYPES=6)
target_compile_options(dynamic_6 PUBLIC -O3 -Winline)
target_link_libraries(dynamic_6 ${Boost_LIBRARIES})
target_link_libraries(dynamic_6 ThrustOMP)

add_executable(dynamic_12 src/dynamic.cpp)
target_compile_definitions(dynamic_12 PUBLIC -DNUM_TYPES=12)
target_compile_options(dynamic_12 PUBLIC -O3 -Winline)
target_link_libraries(dynamic_12 ${Boost_LIBRARIES})
target_link_libraries(dynamic_12 ThrustOMP)

add_executable(dynamic_20 src/dynamic.cpp)
target_compile_definitions(dynamic_20 PUBLIC -DNUM_TYPES=20)
target_compile_options(dynamic_20 PUBLIC -O3 -Winline)
target_link_libraries(dynamic_20 ${Boost_LIBRARIES})
target_link_libraries(dynamic_20 ThrustOMP)