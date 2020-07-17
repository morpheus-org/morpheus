#!/bin/bash

load_cmake()
{
    local __MORPHEUS_PATH=$1
    local __MACHINE=$2
    
    . $__MORPHEUS_PATH/scripts/bash/$__MACHINE/cmake.sh
    
    load_cmake_$__MACHINE
    cmake --version
}