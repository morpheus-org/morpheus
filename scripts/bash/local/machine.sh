#!/bin/bash

configure_scheduler_serial_local()
{
    local __TIME=$1
    local __NAME=$2
    local __FILE=$3
    
    local __FILE_ARGS=""

    for i in "${@:4}"; do
        __FILE_ARGS="$__FILE_ARGS $i"
    done

    local __SCHEDULER_FILE="$__FILE"
    local __SCHEDULER_FILE_ARGS="$__FILE_ARGS"
 
    echo $__SCHEDULER_FILE $__SCHEDULER_FILE_ARGS
}

launch_cmd_serial_local()
{   
    local __EXECUTABLE="$1"
    local __LAUNCH_CMD="$__EXECUTABLE"

    echo $__LAUNCH_CMD
}