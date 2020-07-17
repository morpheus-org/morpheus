#!/bin/bash

set_CC()
{
    local __COMPILER=$1

    if [ "$__COMPILER" == "gcc" ]; then
        local __CC="gcc"
    elif [ "$__COMPILER" == "intel" ]; then
        local __CC="icc"
    fi

    echo "$__CC"
}

set_CXX()
{
    local __COMPILER=$1

    if [ "$__COMPILER" == "gcc" ]; then
        local __CXX="g++"
    elif [ "$__COMPILER" == "intel" ]; then
        local __CXX="icpc"
    fi

    echo "$__CXX"
}

load_compiler()
{
    local __MORPHEUS_PATH=$1
    local __MACHINE=$2
    local __COMPILER=$3
    local __VERSION=$4
    
    . $__MORPHEUS_PATH/scripts/bash/$__MACHINE/compilers.sh

    if [ "$__COMPILER" == "" ]; then
        __COMPILER="gcc"
    fi

    if [ "$__VERSION" == "" ]; then
        __VERSION="6.3.0"
    fi

    load_compiler_$__MACHINE $__COMPILER $__VERSION
    $(set_CC $__COMPILER) --version
    $(set_CXX $__COMPILER) --version
}

