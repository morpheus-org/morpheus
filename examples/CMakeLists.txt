cmake_minimum_required( VERSION 3.5 )

project(morpheus-examples)

message(STATUS "Examples included")

if(NOT Morpheus_BUILD_EXAMPLES)
    message(FATAL_ERROR "Morpheus: Morpheus_BUILD_EXAMPLES not set.")
endif()

# This function takes in an example name and example source and handles setting all of the associated
# properties and linking to build the example
function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    target_include_directories(${EXAMPLE_NAME} PRIVATE "$<BUILD_INTERFACE:${Morpheus_SOURCE_DIR}>")
    target_link_libraries(${EXAMPLE_NAME} morpheus)
    set_target_properties(
    ${EXAMPLE_NAME}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
                RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${Morpheus_BINARY_DIR}/examples>")
    # target_compile_options(${EXAMPLE_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
    #                                             -Wno-error=deprecated-declarations>)
endfunction()

# These examples use the separate compilation
set( SOURCES_IDIOMATIC_EXAMPLES
    version.cpp
    dynamic_resize.cpp
    switch_interface.cpp
    matrix_traits.cpp
    vector_traits.cpp
    morpheus_init.cpp
    vector.cpp
    CooMatrix.cpp
    multiply.cpp
    CsrMatrix.cpp
    omp.cpp
    DenseMatrix.cpp
    DiaMatrix.cpp
    Copy.cpp
    DynamicCopy.cpp
    sorting.cpp
    convert.cpp
    io.cpp
)

string( REPLACE ".cpp" "" BASENAMES_IDIOMATIC_EXAMPLES "${SOURCES_IDIOMATIC_EXAMPLES}" )
set( TARGETS_IDIOMATIC_EXAMPLES ${BASENAMES_IDIOMATIC_EXAMPLES} )

foreach( name ${TARGETS_IDIOMATIC_EXAMPLES} )
    ConfigureExample(${name} ${Morpheus_EXAMPLES_DIR}/${name}.cpp)
endforeach()


