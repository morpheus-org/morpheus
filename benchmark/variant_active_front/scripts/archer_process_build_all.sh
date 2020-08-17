#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

$MORPHEUS_PATH/benchmark/variant_active_front/scripts/process_build.sh "archer" "gcc" "7.3.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/process_build.sh "archer" "gcc" "8.3.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/process_build.sh "archer" "gcc" "9.3.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/process_build.sh "archer" "gcc" "10.1.0"

$MORPHEUS_PATH/benchmark/variant_active_front/scripts/process_build.sh "archer" "intel" "19.5"