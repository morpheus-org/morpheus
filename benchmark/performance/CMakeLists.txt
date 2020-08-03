cmake_minimum_required(VERSION 3.16)
project(morpheus)

set(CMAKE_CXX_STANDARD 14)

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

option(LIKWID "Compiling with likwid" OFF)

if(LIKWID)
    set(LIKWID_FLAG "-DLIKWID_PERFMON")
else()
    set(LIKWID_FLAG "")
endif()

set(Thrust_DIR "${Local_DIR}/thrust/thrust/cmake")
set(Cusp_DIR "${Local_DIR}/cusplibrary-0.5.1")
set(Morpheus_DIR "${Local_DIR}/morpheus")

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)

include_directories(${Cusp_DIR})
include_directories(${Morpheus_DIR})

add_library(likwid SHARED IMPORTED) # or STATIC instead of SHARED
set_target_properties(likwid PROPERTIES
  IMPORTED_LOCATION "/home/software/likwid/5.0.1/lib/liblikwid.so"
  INTERFACE_INCLUDE_DIRECTORIES "/home/software/likwid/5.0.1/include"
)

add_executable(coo src/coo.cpp)
target_compile_options(coo PUBLIC -O3 -pthread ${LIKWID_FLAG})
target_link_libraries(coo ThrustOMP likwid)

add_executable(csr src/csr.cpp)
target_compile_options(csr PUBLIC -O3 -pthread ${LIKWID_FLAG})
target_link_libraries(csr ThrustOMP likwid)

add_executable(hyb src/hyb.cpp)
target_compile_options(hyb PUBLIC -O3 -pthread ${LIKWID_FLAG})
target_link_libraries(hyb ThrustOMP likwid)
