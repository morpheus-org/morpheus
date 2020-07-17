#!/bin/bash

check_supported_machines()
{
    local __MACHINE=$1
    local __SUPPORTED=false
    
    MACHINES=("local" "archer" "cirrus")

    for machine in "${MACHINES[@]}"; do
        if [ "$__MACHINE" == "$machine" ]; then
           __SUPPORTED=true
        fi
    done
    
    echo $__SUPPORTED
}

configure_scheduler_serial()
{
    local __MACHINE=$1
    local __TIME=$2
    local __NAME=$3
    local __FILE=$4
    
    local __FILE_ARGS=""

    for i in "${@:5}"; do
        __FILE_ARGS="$__FILE_ARGS $i"
    done
    
    local __PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    . $__PATH/$__MACHINE/machine.sh

    local __SCHEDULER=$(configure_scheduler_serial_$__MACHINE $__TIME $__NAME $__FILE $__FILE_ARGS)
    
    echo "$__SCHEDULER"
}

launch_cmd_serial()
{
    local __MACHINE=$1
    local __EXECUTABLE=$2

    local __PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    . $__PATH/$__MACHINE/machine.sh

    local __LAUNCH_CMD=$(launch_cmd_serial_$__MACHINE $__EXECUTABLE)

    echo "$__LAUNCH_CMD"
}