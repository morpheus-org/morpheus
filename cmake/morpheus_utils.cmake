macro(GLOBAL_SET VARNAME)
  set(${VARNAME}
      ${ARGN}
      CACHE INTERNAL "")
endmacro()

function(GLOBAL_APPEND VARNAME)
  # We make this a function since we are setting variables and want to use scope
  # to avoid overwriting local variables
  set(TEMP ${${VARNAME}})
  list(APPEND TEMP ${ARGN})
  global_set(${VARNAME} ${TEMP})
endfunction()

macro(APPEND_GLOB VAR)
  file(GLOB LOCAL_TMP_VAR ${ARGN})
  list(APPEND ${VAR} ${LOCAL_TMP_VAR})
endmacro()

function(VERIFY_EMPTY CONTEXT)
  if(${ARGN})
    message(FATAL_ERROR "Unhandled arguments in ${CONTEXT}:\n${ARGN}")
  endif()
endfunction()

macro(MORPHEUS_PACKAGE_DECL)
  set(PACKAGE_NAME Morpheus)
  set(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  set(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endmacro(MORPHEUS_PACKAGE_DECL)

macro(MORPHEUS_PACKAGE_POSTPROCESS)
  include(CMakePackageConfigHelpers)
  if(NOT Morpheus_INSTALL_TESTING)
    configure_package_config_file(
      cmake/MorpheusConfig.cmake.in
      "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)

    write_basic_package_version_file(
      "${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
      VERSION "${Morpheus_VERSION}"
      COMPATIBILITY SameMajorVersion)

    install(FILES "${Morpheus_BINARY_DIR}/MorpheusConfig.cmake"
                  "${Morpheus_BINARY_DIR}/MorpheusConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)

    install(
      EXPORT MorpheusTargets
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus
      NAMESPACE Morpheus::)
  else()
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/MorpheusConfigVersion.cmake"
      VERSION "${Morpheus_VERSION}"
      COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MorpheusConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Morpheus)
  endif()
endmacro(MORPHEUS_PACKAGE_POSTPROCESS)

macro(MORPHEUS_PROCESS_SUBPACKAGES)
  add_subdirectory(core)
  add_subdirectory(examples)
endmacro()

macro(MORPHEUS_SUBPACKAGE NAME)
  set(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
  set(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  set(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endmacro()

macro(MORPHEUS_ADD_TEST_DIRECTORIES)
  if(Morpheus_ENABLE_TESTS)
    foreach(TEST_DIR ${ARGN})
      add_subdirectory(${TEST_DIR})
    endforeach()
  endif()
endmacro()

macro(MORPHEUS_ADD_EXAMPLE_DIRECTORIES)
  if(Morpheus_ENABLE_EXAMPLES)
    foreach(EXAMPLE_DIR ${ARGN})
      add_subdirectory(${EXAMPLE_DIR})
    endforeach()
  endif()
endmacro()

macro(MORPHEUS_ADD_BENCHMARK_DIRECTORIES)
  if(Morpheus_ENABLE_BENCHMARKS)
    foreach(BENCHMARK_DIR ${ARGN})
      add_subdirectory(${BENCHMARK_DIR})
    endforeach()
  endif()
endmacro()

function(morpheus_add_option SUFFIX DEFAULT TYPE DOCSTRING)
  cmake_parse_arguments(
    OPT "" "" "VALID_ENTRIES" # if this is a list variable, the valid values in
                              # the list
    ${ARGN})

  set(CAMEL_NAME Morpheus_${SUFFIX})
  string(TOUPPER ${CAMEL_NAME} UC_NAME)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  set(${CAMEL_NAME}
      ${DEFAULT}
      CACHE ${TYPE} ${DOCSTRING})

  foreach(opt ${MORPHEUS_GIVEN_VARIABLES})
    string(TOUPPER ${opt} OPT_UC)
    if("${OPT_UC}" STREQUAL "${UC_NAME}")
      if(NOT "${opt}" STREQUAL "${CAMEL_NAME}")
        message(
          FATAL_ERROR
            "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies."
        )
      endif()
    endif()
  endforeach()

  # okay, great, we passed the validation test - use the default
  if(DEFINED ${CAMEL_NAME})
    if(OPT_VALID_ENTRIES)
      string(TOUPPER "${OPT_VALID_ENTRIES}" OPT_VALID_ENTRIES_UC)
      foreach(entry ${${CAMEL_NAME}})
        string(TOUPPER ${entry} ENTRY_UC)
        if(NOT ${ENTRY_UC} IN_LIST OPT_VALID_ENTRIES_UC)
          message(
            FATAL_ERROR
              "Given entry ${entry} in list for option ${SUFFIX}. "
              "Valid case-insensitive values are any of ${OPT_VALID_ENTRIES}")
        endif()
      endforeach()
      string(TOUPPER "${${CAMEL_NAME}}" GIVEN_ENTRIES_UC)
      set(${UC_NAME}
          ${GIVEN_ENTRIES_UC}
          PARENT_SCOPE)
    else()
      set(${UC_NAME}
          ${${CAMEL_NAME}}
          PARENT_SCOPE)
    endif()
  else()
    set(${UC_NAME}
        ${DEFAULT}
        PARENT_SCOPE)
  endif()
endfunction()

function(MORPHEUS_ADD_LIBRARY LIBRARY_NAME)
  cmake_parse_arguments(PARSE "ADD_BUILD_OPTIONS" "" "HEADERS" ${ARGN})

  # Forward the headers, we want to know about all headers to make sure they
  # appear correctly in IDEs
  morpheus_internal_add_library(${LIBRARY_NAME} ${PARSE_UNPARSED_ARGUMENTS}
                                HEADERS ${PARSE_HEADERS})
  if(PARSE_ADD_BUILD_OPTIONS)
    morpheus_set_library_properties(${LIBRARY_NAME})
  endif()
endfunction()

function(MORPHEUS_SET_LIBRARY_PROPERTIES LIBRARY_NAME)
  cmake_parse_arguments(PARSE "PLAIN_STYLE" "" "" ${ARGN})

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
    # I can use link options check for CXX linkage using the simple 3.18 way
    target_link_options(${LIBRARY_NAME} PUBLIC
                        $<$<LINK_LANGUAGE:CXX>:${KOKKOS_LINK_OPTIONS}>)
  else()
    # I can use link options just assume CXX linkage
    target_link_options(${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS})
  endif()

  target_compile_options(
    ${LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_COMPILE_OPTIONS}>)

  target_compile_definitions(
    ${LIBRARY_NAME}
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_COMPILE_DEFINITIONS}>)

  target_link_libraries(${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_LIBRARIES})

  if(MORPHEUS_ENABLE_CUDA)
    target_compile_options(
      ${LIBRARY_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_CUDA_OPTIONS}>)
    foreach(OPT ${KOKKOS_CUDAFE_OPTIONS})
      list(APPEND NODEDUP_CUDAFE_OPTIONS -Xcudafe ${OPT})
    endforeach()
    target_compile_options(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_CUDAFE_OPTIONS}>)
  endif()

  if(MORPHEUS_ENABLE_HIP)
    target_compile_options(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_AMDGPU_OPTIONS}>)
  endif()

  list(LENGTH KOKKOS_XCOMPILER_OPTIONS XOPT_LENGTH)
  if(XOPT_LENGTH GREATER 1)
    message(
      FATAL_ERROR
        "CMake deduplication does not allow multiple -Xcompiler flags (${KOKKOS_XCOMPILER_OPTIONS}): will require Kokkos to upgrade to minimum 3.12"
    )
  endif()
  if(KOKKOS_XCOMPILER_OPTIONS)
    set(NODEDUP_XCOMPILER_OPTIONS)
    foreach(OPT ${KOKKOS_XCOMPILER_OPTIONS})
      # I have to do this for now because we can't guarantee 3.12 support I
      # really should do this with the shell option
      list(APPEND NODEDUP_XCOMPILER_OPTIONS -Xcompiler)
      list(APPEND NODEDUP_XCOMPILER_OPTIONS ${OPT})
    endforeach()
    target_compile_options(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_XCOMPILER_OPTIONS}>)
  endif()

  if(KOKKOS_CXX_STANDARD_FEATURE)
    # GREAT! I can do this the right way
    target_compile_features(${LIBRARY_NAME}
                            PUBLIC ${KOKKOS_CXX_STANDARD_FEATURE})
    if(NOT KOKKOS_USE_CXX_EXTENSIONS)
      set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_EXTENSIONS OFF)
    endif()
  else()
    # OH, well, no choice but the wrong way
    target_compile_options(${LIBRARY_NAME} PUBLIC ${KOKKOS_CXX_STANDARD_FLAG})
  endif()
endfunction()

function(MORPHEUS_LIB_TYPE LIB RET)
  get_target_property(PROP ${LIB} TYPE)
  if(${PROP} STREQUAL "INTERFACE_LIBRARY")
    set(${RET}
        "INTERFACE"
        PARENT_SCOPE)
  else()
    set(${RET}
        "PUBLIC"
        PARENT_SCOPE)
  endif()
endfunction()

function(MORPHEUS_INTERNAL_ADD_LIBRARY LIBRARY_NAME)
  cmake_parse_arguments(PARSE "STATIC;SHARED" "" "HEADERS;SOURCES" ${ARGN})

  if(PARSE_HEADERS)
    list(REMOVE_DUPLICATES PARSE_HEADERS)
  endif()
  if(PARSE_SOURCES)
    list(REMOVE_DUPLICATES PARSE_SOURCES)
  endif()

  if(PARSE_STATIC)
    set(LINK_TYPE STATIC)
  endif()

  if(PARSE_SHARED)
    set(LINK_TYPE SHARED)
  endif()

  # MSVC and other platforms want to have the headers included as source files
  # for better dependency detection
  add_library(${LIBRARY_NAME} ${LINK_TYPE} ${PARSE_HEADERS} ${PARSE_SOURCES})

  if(PARSE_SHARED OR BUILD_SHARED_LIBS)
    set_target_properties(
      ${LIBRARY_NAME}
      PROPERTIES VERSION ${MORPHEUS_VERSION}
                 SOVERSION ${Morpheus_VERSION_MAJOR}.${Morpheus_VERSION_MINOR})
  endif()

  morpheus_internal_add_library_install(${LIBRARY_NAME})

  # In case we are building in-tree, add an alias name that matches the install
  # Morpheus:: name
  add_library(Morpheus::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})
endfunction()

macro(MORPHEUS_INTERNAL_ADD_LIBRARY_INSTALL LIBRARY_NAME)
  morpheus_lib_type(${LIBRARY_NAME} INCTYPE)
  target_include_directories(
    ${LIBRARY_NAME} ${INCTYPE}
    $<INSTALL_INTERFACE:${MORPHEUS_HEADER_INSTALL_DIR}>)

  install(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${PACKAGE_NAME})

  install(
    TARGETS ${LIBRARY_NAME}
    EXPORT MorpheusTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  verify_empty(MORPHEUS_ADD_LIBRARY ${PARSE_UNPARSED_ARGUMENTS})
endmacro()

function(MORPHEUS_LIB_INCLUDE_DIRECTORIES TARGET)
  # append to a list for later
  morpheus_lib_type(${TARGET} INCTYPE)
  foreach(DIR ${ARGN})
    target_include_directories(${TARGET} ${INCTYPE} $<BUILD_INTERFACE:${DIR}>)
  endforeach()
endfunction()

macro(MORPHEUS_ADD_DEBUG_OPTION)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # option(HAVE_MORPHEUS_DEBUG "Build Morpheus with debug symbols." ON)
    set(HAVE_MORPHEUS_DEBUG)
  endif()
endmacro()

macro(MORPHEUS_SETUP_BUILD_ENVIRONMENT)
  # set an internal option, if not already set
  set(Morpheus_INSTALL_TESTING
      OFF
      CACHE INTERNAL "Whether to build tests and examples against installation")

  set(Morpheus_ENABLE_TESTS
      OFF
      CACHE INTERNAL "Whether to build tests")

  set(Morpheus_ENABLE_EXAMPLES
      OFF
      CACHE INTERNAL "Whether to build examples")

  set(Morpheus_ENABLE_DOCS
      OFF
      CACHE INTERNAL "Whether to build documentation")

  set(Morpheus_ENABLE_BENCHMARKS
      OFF
      CACHE INTERNAL "Whether to build benchmarks")

  morpheus_add_option(
    ENABLE_ARM_SVE OFF BOOL
    "Whether ARM SVE Intrinsics are enabled. Default: OFF")
  # if(Morpheus_ENABLE_ARM_SVE)
  #   set(MORPHEUS_ENABLE_ARM_SVE ON)
  # endif()
  # global_set(Morpheus_ENABLE_ARM_SVE ${MORPHEUS_ENABLE_ARM_SVE})

  if(Morpheus_INSTALL_TESTING)
    set(Morpheus_ENABLE_TESTS ON)
    set(Morpheus_ENABLE_EXAMPLES ON)
    set(Morpheus_ENABLE_BENCHMARKS ON)
    # We are NOT going build Morpheus but instead look for an installed Morpheus
    # then build examples and tests against that installed Morpheus
    find_package(Morpheus REQUIRED)
    # Still need to figure out which backends
    include(cmake/morpheus_backends.cmake)

    if(Morpheus_ENABLE_TESTS OR Morpheus_ENABLE_EXAMPLES)
      include(cmake/morpheus_gtest.cmake)
    endif()

    if(Morpheus_ENABLE_DOCS)
      include(cmake/morpheus_doxygen.cmake)
    endif()
  else()
    # Regular build, not install testing
    if(NOT MORPHEUS_HAS_PARENT)
      # This is a standalone build
      find_package(Kokkos REQUIRED)
      message(STATUS "Found Kokkos at ${Kokkos_DIR}")
    endif()
    include(cmake/morpheus_backends.cmake)
    include(cmake/morpheus_test_cxx_std.cmake) # TODO: Enforce cxx std17 or
                                               # above

    # If building in debug mode, define the HAVE_MORPHEUS_DEBUG macro.
    morpheus_add_debug_option()

    if(Morpheus_ENABLE_TESTS OR Morpheus_ENABLE_EXAMPLES)
      include(cmake/morpheus_gtest.cmake)
    endif()

    if(Morpheus_ENABLE_DOCS)
      include(cmake/morpheus_doxygen.cmake)
    endif()

    # ==================================================================
    # Enable Third Party Libraries
    # ==================================================================
    include(cmake/morpheus_tpls.cmake)
    # include(cmake/morpheus_features.cmake) # TODO
    include(cmake/kokkos_requirements.cmake)

  endif()
endmacro()

function(MORPHEUS_CONFIGURE_FILE PACKAGE_NAME_CONFIG_FILE)
  # Configure the file
  configure_file(${PACKAGE_SOURCE_DIR}/cmake/${PACKAGE_NAME_CONFIG_FILE}.in
                 ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME_CONFIG_FILE})
endfunction(MORPHEUS_CONFIGURE_FILE)

function(MORPHEUS_INCLUDE_DIRECTORIES)
  cmake_parse_arguments(INC "REQUIRED_DURING_INSTALLATION_TESTING" "" ""
                        ${ARGN})
  include_directories(${INC_UNPARSED_ARGUMENTS})
endfunction()

function(MORPHEUS_ADD_EXECUTABLE_AND_TEST ROOT_NAME)
  cmake_parse_arguments(PARSE "" "" "SOURCES;CATEGORIES;ARGS" ${ARGN})
  verify_empty(MORPHEUS_ADD_EXECUTABLE_AND_TEST ${PARSE_UNPARSED_ARGUMENTS})

  morpheus_add_test_executable(${ROOT_NAME} SOURCES ${PARSE_SOURCES})
  if(PARSE_ARGS)
    set(TEST_NUMBER 0)
    foreach(ARG_STR ${PARSE_ARGS})
      # This is passed as a single string blob to match TriBITS behavior We need
      # this to be turned into a list
      string(REPLACE " " ";" ARG_STR_LIST ${ARG_STR})
      list(APPEND TEST_NAME "${ROOT_NAME}${TEST_NUMBER}")
      math(EXPR TEST_NUMBER "${TEST_NUMBER} + 1")
      morpheus_add_test(
        NAME
        ${TEST_NAME}
        EXE
        ${ROOT_NAME}
        FAIL_REGULAR_EXPRESSION
        "  FAILED  "
        ARGS
        ${ARG_STR_LIST})
    endforeach()
  else()
    morpheus_add_test(NAME ${ROOT_NAME} EXE ${ROOT_NAME}
                      FAIL_REGULAR_EXPRESSION "  FAILED  ")
  endif()
endfunction()

macro(MORPHEUS_ADD_TEST_EXECUTABLE ROOT_NAME)
  cmake_parse_arguments(PARSE "" "" "SOURCES" ${ARGN})
  morpheus_add_executable(
    ${ROOT_NAME} SOURCES ${PARSE_SOURCES} ${PARSE_UNPARSED_ARGUMENTS}
    TESTONLYLIBS morpheus_gtest)
  set(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
endmacro()

function(MORPHEUS_ADD_TEST)
  cmake_parse_arguments(
    TEST "WILL_FAIL;"
    "FAIL_REGULAR_EXPRESSION;PASS_REGULAR_EXPRESSION;EXE;NAME;TOOL"
    "CATEGORIES;ARGS" ${ARGN})
  if(TEST_EXE)
    set(EXE_ROOT ${TEST_EXE})
  else()
    set(EXE_ROOT ${TEST_NAME})
  endif()
  # Prepend package name to the test name These should be the full target name
  set(TEST_NAME ${PACKAGE_NAME}_${TEST_NAME})
  set(EXE ${PACKAGE_NAME}_${EXE_ROOT})
  if(WIN32)
    add_test(
      NAME ${TEST_NAME}
      WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
      COMMAND ${EXE}${CMAKE_EXECUTABLE_SUFFIX} ${TEST_ARGS})
  else()
    add_test(NAME ${TEST_NAME} COMMAND ${EXE} ${TEST_ARGS})
  endif()
  if(TEST_WILL_FAIL)
    set_tests_properties(${TEST_NAME} PROPERTIES WILL_FAIL ${TEST_WILL_FAIL})
  endif()
  if(TEST_FAIL_REGULAR_EXPRESSION)
    set_tests_properties(
      ${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION
                              ${TEST_FAIL_REGULAR_EXPRESSION})
  endif()
  if(TEST_PASS_REGULAR_EXPRESSION)
    set_tests_properties(
      ${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION
                              ${TEST_PASS_REGULAR_EXPRESSION})
  endif()
  verify_empty(MORPHEUS_ADD_TEST ${TEST_UNPARSED_ARGUMENTS})
endfunction()

function(MORPHEUS_ADD_EXECUTABLE ROOT_NAME)
  cmake_parse_arguments(PARSE "TESTONLY" "" "SOURCES;TESTONLYLIBS" ${ARGN})

  set(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
  add_executable(${EXE_NAME} ${PARSE_SOURCES})
  if(PARSE_TESTONLYLIBS)
    target_link_libraries(${EXE_NAME} PRIVATE ${PARSE_TESTONLYLIBS})
  endif()
  verify_empty(MORPHEUS_ADD_EXECUTABLE ${PARSE_UNPARSED_ARGUMENTS})
  # All executables must link to all the morpheus targets This is just private
  # linkage because exe is final
  target_link_libraries(${EXE_NAME} PRIVATE Morpheus::morpheus)
endfunction()
