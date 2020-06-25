#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../../matrix

BUILD_PATH="$SCRIPT_PATH/../build"
RESULTS_PATH="$SCRIPT_PATH/../results"
SUBMIT_FILE="$SCRIPT_PATH/submit.pbs"
VERSIONS=("cusp" "dynamic_1" "dynamic_6" "dynamic_12" "dynamic_20")
REPS=5
SPMV_ITER=10

# Archer stuff
ACCOUNT="e609"
TIME="walltime=06:00:00"
SELECT="select=1:ncpus=24"
RESOURCES="$SELECT,$TIME"
QSUB="qsub -A $ACCOUNT -l ${RESOURCES}"


mkdir -p "$RESULTS_PATH"

for version in "${VERSIONS[@]}"
do
  progress="$RESULTS_PATH/progress"_"$version.txt"
  echo "Starting version $version" 2>&1 | tee "$progress"

  BINARY="$BUILD_PATH/$version"

  $QSUB -N "variant_$version" \
        -v SCRIPT_PATH="$SCRIPT_PATH",MATRIX_DIR="$MATRIX_DIR",RESULTS_PATH="$RESULTS_PATH",PROGRESS_FILE="$progress",VERSION="$version",BINARY="$BINARY",REPS="$REPS",SPMV_ITER="$SPMV_ITER" \
        $SUBMIT_FILE

done