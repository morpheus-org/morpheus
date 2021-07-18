message(STATUS "Core Examples are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

morpheus_add_executable_and_test(Examples_Init SOURCES Examples_Init.cpp)
morpheus_add_executable_and_test(Examples_Version SOURCES Examples_Version.cpp)
morpheus_add_executable_and_test(Examples_Multiply SOURCES
                                 Examples_Multiply.cpp)
morpheus_add_executable_and_test(Examples_DynamicResize SOURCES
                                 Examples_DynamicResize.cpp)
morpheus_add_executable_and_test(Examples_DiaMatrix SOURCES
                                 Examples_DiaMatrix.cpp)
