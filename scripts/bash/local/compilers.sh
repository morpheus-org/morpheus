#!/bin/bash

load_compiler_local()
{
    local __COMPILER=$1
    local __VERSION=$2

    if [ -f $HOME/.bash_profile ]; then
        . $HOME/.bash_profile
    fi

    if [ "$__COMPILER" == "gcc" ]; then
        if [ "$__VERSION" == "10.1.0" ]; then
            load_gcc $__VERSION
        else
            echo "GCC $__VERSION is not supported."
            exit -1
        fi
    elif [ "$__COMPILER" == "intel" ]; then
        if [ "$__VERSION" == "20" ]; then
            load_intel $__VERSION
        else
            echo "Intel $__VERSION is not supported."
            exit -1
        fi
    else
        echo "$__COMPILER is an invalid compiler."
        echo "Valid compiler options are gcc or intel."
        exit -1
    fi
}