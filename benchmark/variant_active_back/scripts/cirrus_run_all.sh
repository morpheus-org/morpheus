#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "gcc" "8.2.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "gcc" "8.3.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "gcc" "9.3.0" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "gcc" "10.1.0" "20" "500" "06:00:00" "standard"

$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "intel" "19.5" "20" "500" "06:00:00" "standard"
$MORPHEUS_PATH/benchmark/variant_active_back/scripts/run.sh "cirrus" "intel" "20.1" "20" "500" "06:00:00" "standard"