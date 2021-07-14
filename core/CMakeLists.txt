morpheus_subpackage(Core)

if(NOT Morpheus_INSTALL_TESTING)
  add_subdirectory(src)
endif()

morpheus_add_test_directories(tests)
morpheus_add_test_directories(benchmarks)
morpheus_add_example_directories(examples)
