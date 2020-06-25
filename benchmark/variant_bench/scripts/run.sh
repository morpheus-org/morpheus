#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../../matrix

MACHINE="$1"

if [ "$MACHINE" = "local" ]; then
  echo "Running locally"
  SUBMIT_FILE="$SCRIPT_PATH/submit.sh"
elif [ "$MACHINE" = "archer" ]; then
  echo "Running on ARCHER"
  # Archer stuff
  ACCOUNT="e609"
  TIME="walltime=06:00:00"
  SELECT="select=1:ncpus=24"
  RESOURCES="$SELECT,$TIME"
  QSUB="qsub -A $ACCOUNT -l ${RESOURCES}"
  SUBMIT_FILE="$SCRIPT_PATH/submit_archer.pbs"
elif [ "$MACHINE" = "cirrus" ]; then
  echo "Running on CIRRUS"
  # Cirrus stuff
  ACCOUNT="dc111"
  TIME="06:00:00"
  RESOURCES="--exclusive --nodes=1 --tasks-per-node=36 --cpus-per-task=1"
  SBATCH="sbatch --job-name=$ACCOUNT --time=$TIME $RESOURCES --partition=standard --qos=standard"
  SUBMIT_FILE="$SCRIPT_PATH/submit_cirrus.slurm"
else
  echo "Invalid inpug argument."
  echo "Usage:"
  echo -e "\t/path/to/script/build.sh [local|archer|cirrus]"
  exit -1
fi

BUILD_PATH="$SCRIPT_PATH/../build"
RESULTS_PATH="$SCRIPT_PATH/../results/$MACHINE"
VERSIONS=("cusp" "dynamic_1" "dynamic_6" "dynamic_12" "dynamic_20")
REPS=5
SPMV_ITER=100

mkdir -p "$RESULTS_PATH"

for version in "${VERSIONS[@]}"
do
  progress="$RESULTS_PATH/progress"_"$version.txt"
  echo "Starting version $version" 2>&1 | tee "$progress"

  BINARY="$BUILD_PATH/$version"
  if [ "$MACHINE" = "archer" ]; then
    $QSUB -N "variant_$version" \
          -v SCRIPT_PATH="$SCRIPT_PATH",MATRIX_DIR="$MATRIX_DIR",RESULTS_PATH="$RESULTS_PATH",PROGRESS_FILE="$progress",VERSION="$version",BINARY="$BINARY",REPS="$REPS",SPMV_ITER="$SPMV_ITER" \
          $SUBMIT_FILE
  elif [ "$MACHINE" = "cirrus" ]; then
    $SBATCH $SUBMIT_FILE "$SCRIPT_PATH" "$MATRIX_DIR" "$RESULTS_PATH" "$progress" "$version" "$BINARY" "$REPS" "$SPMV_ITER"
  else
    $SUBMIT_FILE "$SCRIPT_PATH" "$MATRIX_DIR" "$RESULTS_PATH" "$progress" "$version" "$BINARY" "$REPS" "$SPMV_ITER"
  fi

done
