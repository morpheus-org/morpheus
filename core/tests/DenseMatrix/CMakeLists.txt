set(Fmt DenseMatrix)
string(TOUPPER ${Fmt} FORMAT)
string(TOLOWER ${Fmt} fmtdir)

foreach(Tag Serial;OpenMP;Cuda;HIP)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)

  if(MORPHEUS_ENABLE_${DEVICE})
    set(testdir ${CMAKE_CURRENT_BINARY_DIR}/${fmtdir}/${dir})
    file(MAKE_DIRECTORY ${testdir})

    foreach(Name MirrorContainers DeepCopy)
      set(file ${testdir}/Test${Fmt}_${Tag}_${Name}.cpp)
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake
      # runs.
      file(
        WRITE ${testdir}/dummy.cpp
        "#include <Test${Fmt}_Format.hpp>\n"
        "#include <Test${Tag}_Category.hpp>\n"
        "#include <Test${Fmt}_Utils.hpp>\n"
        "#define TESTSUITE_NAME ${Fmt}${Tag}\n"
        "#include <${Fmt}/Test_${Name}.hpp>\n")
      configure_file(${testdir}/dummy.cpp ${file})
      global_append(${Tag}_SOURCES ${file})
    endforeach()
  endif()
endforeach()
