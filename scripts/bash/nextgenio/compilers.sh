#!/bin/bash

load_compiler_nextgenio()
{
    local __COMPILER=$1
    local __VERSION=$2

    if [ -f $HOME/.bashrc ]; then
        . $HOME/.bashrc
    fi

    if [ "$__COMPILER" == "gcc" ]; then
        if [ "$__VERSION" == "7.3.0" ]; then
            module load "gnu7/$__VERSION"
        elif [ "$__VERSION" == "8.3.0" ]; then
            module load "gnu8/$__VERSION"
        else
            echo "GCC $__VERSION is not supported."
            exit -1
        fi
    elif [ "$__COMPILER" == "intel" ]; then
        if [ "$__VERSION" == "19.0" ]; then
            module load intel/19.0.3.199
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