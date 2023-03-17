message(STATUS "Setting up Morpheus_Gtest library")

find_package(GTest CONFIG)
if(NOT GTest_FOUND)
  message(STATUS "find_package could not find GTest - Downloading GTest")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "" FORCE)
  set(gtest_disable_pthreads
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_library(morpheus_gtest ALIAS gtest_main)
else()
  set_target_properties(GTest::gtest PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(morpheus_gtest ALIAS GTest::gtest)
endif()

<<<<<<< HEAD
=======
morpheus_add_option(
  ENABLE_RAPID_TESTING OFF BOOL
  "Whether rapid testing is enabled during unit tests. Default: OFF")
if(Morpheus_ENABLE_RAPID_TESTING)
  set(MORPHEUS_RAPID_TESTING ON)
endif()
global_set(Morpheus_ENABLE_RAPID_TESTING ${MORPHEUS_RAPID_TESTING})

>>>>>>> new-formats
message(STATUS "Morpheus_Gtest Library configured.")
