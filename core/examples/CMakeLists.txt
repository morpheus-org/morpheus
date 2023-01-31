message(STATUS "Core Examples are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

morpheus_add_executable_and_test(Examples_SpMV SOURCES Examples_SpMV.cpp)
