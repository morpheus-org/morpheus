if(MPARK_VARIANT_LIBRARY_DIRS AND MPARK_VARIANT_LIBRARIES)
  morpheus_find_imported(
    MPARK_VARIANT INTERFACE LIBRARIES ${MPARK_VARIANT_LIBRARIES} LIBRARY_PATHS
    ${MPARK_VARIANT_LIBRARY_DIRS})
elseif(MPARK_VARIANT_LIBRARIES)
  morpheus_find_imported(MPARK_VARIANT INTERFACE LIBRARIES
                         ${MPARK_VARIANT_LIBRARIES})
elseif(MPARK_VARIANT_LIBRARY_DIRS)
  morpheus_find_imported(MPARK_VARIANT INTERFACE LIBRARIES mpark_variant
                         LIBRARY_PATHS ${MPARK_VARIANT_LIBRARY_DIRS})
else()
  find_package(MPARK_VARIANT REQUIRED)
  morpheus_create_imported_tpl(
    MPARK_VARIANT INTERFACE LINK_LIBRARIES ${MPARK_VARIANT_LIBRARIES} INCLUDES
    ${MPARK_VARIANT_INCLUDE_DIRS})
endif()
