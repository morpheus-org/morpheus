cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 17)

option(LOCAL "Compiling locally" OFF)
option(ARCHER "Compiling on archer" OFF)
option(CIRRUS "Compiling on cirrus" OFF)
option(FULHAME "Compiling on fulhame" OFF)
option(NEXTGENIO "Compiling on NextGenIO" OFF)

if(LOCAL)
    # Local directories
    set(Local_DIR "/Users/cstyl/Desktop/Projects")
elseif(ARCHER)
    # Archer directories
    set(Local_DIR "/work/e609/e609/cstyl")
elseif(CIRRUS)
    # Cirrus directories
    set(Local_DIR "/lustre/home/dc111/cstyl")
elseif(FULHAME)
    # Fulhame directories
    set(Local_DIR "/home/fh01/fh01/cstyl")
elseif(NEXTGENIO)
    # NextgenIO directories
    set(Local_DIR "/home/nx01/nx01/cstyl")
else()
    message(FATAL_ERROR "Local Directory not found")
endif()

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


add_executable(static src/static.cpp)
target_compile_options(static PUBLIC -O3)
target_link_libraries(static ThrustOMP)

add_executable(dynamic_01 src/dynamic.cpp)
target_compile_definitions(dynamic_01 PUBLIC -DNUM_TYPES=1)
target_compile_options(dynamic_01 PUBLIC -O3)
target_link_libraries(dynamic_01 ThrustOMP)

add_executable(dynamic_06 src/dynamic.cpp)
target_compile_definitions(dynamic_06 PUBLIC -DNUM_TYPES=6)
target_compile_options(dynamic_06 PUBLIC -O3)
target_link_libraries(dynamic_06 ThrustOMP)

add_executable(dynamic_12 src/dynamic.cpp)
target_compile_definitions(dynamic_12 PUBLIC -DNUM_TYPES=12)
target_compile_options(dynamic_12 PUBLIC -O3)
target_link_libraries(dynamic_12 ThrustOMP)

add_executable(dynamic_20 src/dynamic.cpp)
target_compile_definitions(dynamic_20 PUBLIC -DNUM_TYPES=20)
target_compile_options(dynamic_20 PUBLIC -O3)
target_link_libraries(dynamic_20 ThrustOMP)

add_executable(dynamic_01_boost src/dynamic.cpp)
target_compile_definitions(dynamic_01_boost PUBLIC -DNUM_TYPES=1 -DBOOST_VARIANT)
target_compile_options(dynamic_01_boost PUBLIC -O3)
target_link_libraries(dynamic_01_boost ${Boost_LIBRARIES})
target_link_libraries(dynamic_01_boost ThrustOMP)

add_executable(dynamic_06_boost src/dynamic.cpp)
target_compile_definitions(dynamic_06_boost PUBLIC -DNUM_TYPES=6 -DBOOST_VARIANT)
target_compile_options(dynamic_06_boost PUBLIC -O3)
target_link_libraries(dynamic_06_boost ${Boost_LIBRARIES})
target_link_libraries(dynamic_06_boost ThrustOMP)

add_executable(dynamic_12_boost src/dynamic.cpp)
target_compile_definitions(dynamic_12_boost PUBLIC -DNUM_TYPES=12 -DBOOST_VARIANT)
target_compile_options(dynamic_12_boost PUBLIC -O3)
target_link_libraries(dynamic_12_boost ${Boost_LIBRARIES})
target_link_libraries(dynamic_12_boost ThrustOMP)

add_executable(dynamic_20_boost src/dynamic.cpp)
target_compile_definitions(dynamic_20_boost PUBLIC -DNUM_TYPES=20 -DBOOST_VARIANT)
target_compile_options(dynamic_20_boost PUBLIC -O3)
target_link_libraries(dynamic_20_boost ${Boost_LIBRARIES})
target_link_libraries(dynamic_20_boost ThrustOMP)

add_executable(static_O2 src/static.cpp)
target_compile_options(static_O2 PUBLIC -O2)
target_link_libraries(static_O2 ThrustOMP)

add_executable(dynamic_01_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_01_O2 PUBLIC -DNUM_TYPES=1)
target_compile_options(dynamic_01_O2 PUBLIC -O2)
target_link_libraries(dynamic_01_O2 ThrustOMP)

add_executable(dynamic_06_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_06_O2 PUBLIC -DNUM_TYPES=6)
target_compile_options(dynamic_06_O2 PUBLIC -O2)
target_link_libraries(dynamic_06_O2 ThrustOMP)

add_executable(dynamic_12_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_12_O2 PUBLIC -DNUM_TYPES=12)
target_compile_options(dynamic_12_O2 PUBLIC -O2)
target_link_libraries(dynamic_12_O2 ThrustOMP)

add_executable(dynamic_20_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_20_O2 PUBLIC -DNUM_TYPES=20)
target_compile_options(dynamic_20_O2 PUBLIC -O2)
target_link_libraries(dynamic_20_O2 ThrustOMP)

add_executable(dynamic_01_boost_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_01_boost_O2 PUBLIC -DNUM_TYPES=1 -DBOOST_VARIANT)
target_compile_options(dynamic_01_boost_O2 PUBLIC -O2)
target_link_libraries(dynamic_01_boost_O2 ${Boost_LIBRARIES})
target_link_libraries(dynamic_01_boost_O2 ThrustOMP)

add_executable(dynamic_06_boost_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_06_boost_O2 PUBLIC -DNUM_TYPES=6 -DBOOST_VARIANT)
target_compile_options(dynamic_06_boost_O2 PUBLIC -O2)
target_link_libraries(dynamic_06_boost_O2 ${Boost_LIBRARIES})
target_link_libraries(dynamic_06_boost_O2 ThrustOMP)

add_executable(dynamic_12_boost_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_12_boost_O2 PUBLIC -DNUM_TYPES=12 -DBOOST_VARIANT)
target_compile_options(dynamic_12_boost_O2 PUBLIC -O2)
target_link_libraries(dynamic_12_boost_O2 ${Boost_LIBRARIES})
target_link_libraries(dynamic_12_boost_O2 ThrustOMP)

add_executable(dynamic_20_boost_O2 src/dynamic.cpp)
target_compile_definitions(dynamic_20_boost_O2 PUBLIC -DNUM_TYPES=20 -DBOOST_VARIANT)
target_compile_options(dynamic_20_boost_O2 PUBLIC -O2)
target_link_libraries(dynamic_20_boost_O2 ${Boost_LIBRARIES})
target_link_libraries(dynamic_20_boost_O2 ThrustOMP)