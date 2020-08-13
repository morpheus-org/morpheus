#!/bin/bash

configure_scheduler_serial_fulhame()
{
    local __QUEUE=$1
    local __TIME=$2
    local __NAME=$3
    local __FILE=$4
    
    local __FILE_ARGS=""

    for i in "${@:5}"; do
        __FILE_ARGS="$__FILE_ARGS $i"
    done

    local __ACCOUNT="fh01"
    local __RESOURCES="--time=$__TIME --exclusive --nodes=1 --cpus-per-task=1"
    local __SYSTEM="--partition=workq"

    local __SCHEDULER="sbatch"
    local __SCHEDULER_ARGS="--account=$__ACCOUNT --job-name=$__NAME $__RESOURCES $__SYSTEM"
    local __SCHEDULER_FILE="$__FILE"
    local __SCHEDULER_FILE_ARGS="$__FILE_ARGS"
 
    echo $__SCHEDULER $__SCHEDULER_ARGS $__SCHEDULER_FILE $__SCHEDULER_FILE_ARGS
}

launch_cmd_serial_fulhame()
{
    local __EXECUTABLE="$1"
    local __LAUNCH_CMD="srun -n 1 --ntasks=1 $__EXECUTABLE"

    echo $__LAUNCH_CMD
}