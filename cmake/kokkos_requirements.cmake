if(NOT MORPHEUS_HAS_PARENT)

  if(DEFINED Morpheus_REQUIRE_DEVICES)
    string(REPLACE "," ";" REQUIRE_DEVICES "${Morpheus_REQUIRE_DEVICES}")
    kokkos_check(DEVICES ${REQUIRE_DEVICES})
  endif()

  if(DEFINED Morpheus_REQUIRE_OPTIONS)
    string(REPLACE "," ";" REQUIRE_OPTIONS "${Morpheus_REQUIRE_OPTIONS}")
    kokkos_check(OPTIONS ${REQUIRE_OPTIONS})
  endif()

  if(DEFINED Morpheus_REQUIRE_ARCH)
    string(REPLACE "," ";" REQUIRE_ARCH "${Morpheus_REQUIRE_ARCH}")
    kokkos_check(ARCH ${REQUIRE_ARCH})
  endif()

  if(DEFINED Morpheus_REQUIRE_TPLS)
    string(REPLACE "," ";" REQUIRE_TPLS "${Morpheus_REQUIRE_TPLS}")
    kokkos_check(TPLS ${REQUIRE_TPLS})
  endif()

endif()
