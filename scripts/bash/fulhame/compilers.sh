#!/bin/bash

load_compiler_fulhame()
{
    local __COMPILER=$1
    local __VERSION=$2

    if [ -f $HOME/.bashrc ]; then
        . $HOME/.bashrc
    fi

    if [ "$__COMPILER" == "gcc" ]; then
        if [ "$__VERSION" == "9.2.0" ] || [ "$__VERSION" == "10.1" ]; then
            module load "Generic-AArch64/SUSE/12/$__COMPILER/$__VERSION"
        else
            echo "GCC $__VERSION is not supported."
            exit -1
        fi
    elif [ "$__COMPILER" == "arm" ]; then
        if [ "$__VERSION" == "20.0" ]; then
            module load "Generic-AArch64/SUSE/12/arm-linux-compiler/$__VERSION"
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