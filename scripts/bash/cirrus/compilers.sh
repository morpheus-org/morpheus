#!/bin/bash

load_compiler_cirrus()
{
    local __COMPILER=$1
    local __VERSION=$2

    if [ -f $HOME/.bashrc ]; then
        . $HOME/.bashrc
    fi

    if [ "$__COMPILER" == "gcc" ]; then
        if [ "$__VERSION" == "6.3.0" ] || [ "$__VERSION" == "8.2.0" ]; then
            module load "$__COMPILER/$__VERSION"
        elif [ "$__VERSION" == "8.3.0" ] || [ "$__VERSION" == "9.3.0" ] || [ "$__VERSION" == "10.1.0" ]; then
            load_gcc $__VERSION
        else
            echo "GCC $__VERSION is not supported."
            exit -1
        fi
    elif [ "$__COMPILER" == "intel" ]; then
        if [ "$__VERSION" == "19.5" ] || [ "$__VERSION" == "20.1" ]; then
            load_intel "$__VERSION"
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