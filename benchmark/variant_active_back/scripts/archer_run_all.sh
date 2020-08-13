#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "archer" "gcc" "7.3.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "archer" "gcc" "8.3.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "archer" "gcc" "9.3.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "archer" "gcc" "10.1.0" "20" "500" "06:00:00" "standard"

$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "archer" "intel" "19.5" "20" "500" "06:00:00" "standard"