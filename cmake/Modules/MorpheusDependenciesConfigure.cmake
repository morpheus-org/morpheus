include(cmake/Modules/DownloadProject.cmake)

function(morpheus_configure_thrust) 
    set(Morpheus_MIN_VERSION_Thrust 1.10.0 PARENT_SCOPE)

    # If Thrust_DIR is not set download Thrust
    if(NOT Thrust_DIR)
        download_project(
            PROJ                thrust
            GIT_REPOSITORY      https://github.com/NVIDIA/thrust.git
            GIT_TAG             1.10.0
            SOURCE_DIR          ${Morpheus_SOURCE_DIR}/thirdparty/thrust
            UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
        )
        
        set(Thrust_DIR ${Morpheus_SOURCE_DIR}/thirdparty/thrust/thrust/cmake PARENT_SCOPE)
    endif()
    
endfunction()
