#!/bin/bash

configure_scheduler_serial_archer()
{
    local __TIME=$1
    local __QUEUE=$2
    local __NAME=$3
    local __FILE=$4
    local __FILE_ARGS=""

    for i in "${@:5}"; do
        __FILE_ARGS="$__FILE_ARGS $i"
    done

    local __ACCOUNT="e609"
    local __RESOURCES="select=1:ncpus=24,walltime=$__TIME"

    local __SCHEDULER="qsub"
    local __SCHEDULER_ARGS="-q $__QUEUE -A $__ACCOUNT -l $__RESOURCES -N $__NAME"
    local __SCHEDULER_FILE="-- $__FILE"
    local __SCHEDULER_FILE_ARGS="$__FILE_ARGS"

    echo $__SCHEDULER $__SCHEDULER_ARGS $__SCHEDULER_FILE $__SCHEDULER_FILE_ARGS
}

launch_cmd_serial_archer()
{
    local __EXECUTABLE="$1"
    local __LAUNCH_CMD="aprun -n 1 $__EXECUTABLE"

    echo $__LAUNCH_CMD
}