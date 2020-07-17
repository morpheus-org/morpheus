#!/bin/bash

load_cmake()
{
    local  __MACHINE=$1
    local __PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    
    . $__PATH/$__MACHINE/cmake.sh
    
    load_cmake_$__MACHINE
    cmake --version
}