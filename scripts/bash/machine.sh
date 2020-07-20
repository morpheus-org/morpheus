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
    local __MORPHEUS_PATH=$1
    local __MACHINE=$2
    local __QUEUE=$3
    local __TIME=$4
    local __NAME=$5
    local __FILE=$6
    
    local __FILE_ARGS=""

    for i in "${@:6}"; do
        __FILE_ARGS="$__FILE_ARGS $i"
    done
    
    . $__MORPHEUS_PATH/scripts/bash/$__MACHINE/machine.sh

    local __SCHEDULER=$(configure_scheduler_serial_$__MACHINE $__QUEUE $__TIME $__NAME $__FILE $__FILE_ARGS)
    
    echo "$__SCHEDULER"
}

launch_cmd_serial()
{
    local __MORPHEUS_PATH=$1
    local __MACHINE=$2
    local __EXECUTABLE=$3

    . $__MORPHEUS_PATH/scripts/bash/$__MACHINE/machine.sh

    local __LAUNCH_CMD=$(launch_cmd_serial_$__MACHINE $__EXECUTABLE)

    echo "$__LAUNCH_CMD"
}