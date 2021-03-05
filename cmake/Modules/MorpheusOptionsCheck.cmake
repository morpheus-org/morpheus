function(morpheus_select_programming_model MODEL)
    set(Morpheus_THRUST OFF PARENT_SCOPE)
    set(Morpheus_KOKKOS OFF PARENT_SCOPE)
    set(Morpheus_SYCL OFF PARENT_SCOPE)
    
    string( TOUPPER "${MODEL}" MODEL_UPPER )    # Convert to upper-case for the comparison
    
    if(MODEL_UPPER)
        if(MODEL_UPPER STREQUAL "THRUST")
            set(Morpheus_THRUST ON PARENT_SCOPE)
        elseif(MODEL_UPPER STREQUAL "KOKKOS")
            set(Morpheus_KOKKOS ON PARENT_SCOPE)
            message(WARNING "TODO: ${MODEL_UPPER} Programming Model option has not been implemented yet.")
        elseif(MODEL_UPPER STREQUAL "SYCL")
            set(Morpheus_SYCL ON PARENT_SCOPE)
            message(WARNING "TODO: ${MODEL_UPPER} Programming Model option has not been implemented yet.")
        else()
            message(FATAL_ERROR "Invalid Programming Model option")
        endif()

        message(STATUS "Morpheus Programming model used: ${MODEL}")
    endif()

    # mark_as_advanced(Morpheus_THRUST Morpheus_KOKKOS Morpheus_SYCL)
endfunction()

function(morpheus_select_backend BACKEND)
    set(Morpheus_CPP OFF PARENT_SCOPE)
    set(Morpheus_OMP OFF PARENT_SCOPE)
    set(Morpheus_CUDA OFF PARENT_SCOPE)
    set(Morpheus_ROCM OFF PARENT_SCOPE)
    
    string( TOUPPER "${BACKEND}" BACKEND_UPPER )    # Convert to upper-case for the comparison

    if(BACKEND_UPPER)
        if(BACKEND_UPPER STREQUAL "CPP")
            set(Morpheus_CPP ON PARENT_SCOPE)
        elseif(BACKEND_UPPER STREQUAL "OMP")
            set(Morpheus_OMP ON PARENT_SCOPE)
            message(WARNING "TODO: ${BACKEND_UPPER} backend option has not been implemented yet.")
        elseif(BACKEND_UPPER STREQUAL "CUDA" PARENT_SCOPE)
            set(Morpheus_CUDA ON)
            message(WARNING "TODO: ${BACKEND_UPPER} backend option has not been implemented yet.")
        elseif(BACKEND_UPPER STREQUAL "ROCM" PARENT_SCOPE)
            set(Morpheus_ROCM ON)
            message(WARNING "TODO: ${BACKEND_UPPER} backend option has not been implemented yet.")
        else()
            message(FATAL_ERROR "BACKEND_UPPER = ${BACKEND_UPPER} is an invalid backend option.")
        endif()
    endif()

    message(STATUS "Morpheus Backend used: ${BACKEND_UPPER}")
    # mark_as_advanced(Morpheus_CPP Morpheus_OMP Morpheus_CUDA Morpheus_ROCM)
endfunction()

