#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus gcc 8.2.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus gcc 8.3.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus gcc 9.3.0"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus gcc 10.1.0"

$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus intel 19.5"
$MORPHEUS_PATH/benchmark/variant_active_front/scripts/build.sh "cirrus intel 20.1"