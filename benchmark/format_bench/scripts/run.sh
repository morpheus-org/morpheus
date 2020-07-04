#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../../matrix

MACHINE="$1"
COMPILER="$2"
COMP_VERSION="$3"
REPS="$4"
SPMV_ITER="$5"

if [ "$MACHINE" = "local" ]; then
  echo "Running locally"
  SUBMIT_FILE="$SCRIPT_PATH/submit.sh"
elif [ "$MACHINE" = "archer" ]; then
  echo "Running on ARCHER"
  # Archer stuff
  ACCOUNT="e609"
  TIME="walltime=12:00:00"
  SELECT="select=1:ncpus=24"
  RESOURCES="$SELECT,$TIME"
  QSUB="qsub -A $ACCOUNT -l ${RESOURCES}"
  SUBMIT_FILE="$SCRIPT_PATH/submit_archer.pbs"
elif [ "$MACHINE" = "cirrus" ]; then
  echo "Running on CIRRUS"
  # Cirrus stuff
  ACCOUNT="dc111"
  TIME="12:00:00"
#  RESOURCES="--exclusive --nodes=1 --tasks-per-node=36 --cpus-per-task=1"
  RESOURCES="--exclusive --nodes=1 --cpus-per-task=1"
  echo "RESOURCES = $RESOURCES"
  SBATCH="sbatch --job-name=$ACCOUNT --time=$TIME $RESOURCES --partition=standard --qos=standard"
  SUBMIT_FILE="$SCRIPT_PATH/submit_cirrus.slurm"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/run.sh [local|archer|cirrus] [compiler] [version] [reps] [iter]\n\n"
  echo -e "\t\t local|archer|cirrus: Select at which machine the code is running on."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  echo -e "\t\t reps: Number of experiment repetitons (default is 20)."
  echo -e "\t\t iter: Number of repetitions the spMv multiplication is repeated (default is 100)."
  exit -1
fi

if [ "$COMPILER" == "" ]; then
  COMPILER="gcc"
fi

if [ "$COMP_VERSION" == "" ]; then
  COMP_VERSION="6.3.0"
fi

if [ "$REPS" == "" ]; then
  REPS=20
fi

if [ "$SPMV_ITER" == "" ]; then
  SPMV_ITER=100
fi

echo "Parameters:"
echo -e "\t MACHINE = $MACHINE"
echo -e "\t REPETITIONS = $REPS"
echo -e "\t SPMV Iterations = $SPMV_ITER"
echo -e "\t COMPILER = $COMPILER"
echo -e "\t VERSION = $COMP_VERSION"


BUILD_PATH="$SCRIPT_PATH/../build/$COMPILER/$COMP_VERSION"
RESULTS_PATH="$SCRIPT_PATH/../results/$MACHINE/$COMPILER/$COMP_VERSION"
#FORMATS=("coo", "csr", "dia", "ell", "hyb")
FORMATS=("coo" "csr" "hyb") # for now these do not require fill in
mkdir -p "$RESULTS_PATH"

for format in "${FORMATS[@]}"
do
  progress="$RESULTS_PATH/progress"_"$format.txt"
  echo "Starting format $format" 2>&1 | tee "$progress"

  BINARY="$BUILD_PATH/dynamic_selection"
  if [ "$MACHINE" = "archer" ]; then
    $QSUB -N "dynamic_selection" \
          -v SCRIPT_PATH="$SCRIPT_PATH",MATRIX_DIR="$MATRIX_DIR",RESULTS_PATH="$RESULTS_PATH",PROGRESS_FILE="$progress",BINARY="$BINARY",REPS="$REPS",SPMV_ITER="$SPMV_ITER",FORMAT="$format" \
          $SUBMIT_FILE
  elif [ "$MACHINE" = "cirrus" ]; then
    $SBATCH $SUBMIT_FILE "$SCRIPT_PATH" "$MATRIX_DIR" "$RESULTS_PATH" "$progress" "$BINARY" "$REPS" "$SPMV_ITER" "$format"
  else
    $SUBMIT_FILE "$SCRIPT_PATH" "$MATRIX_DIR" "$RESULTS_PATH" "$progress" "$BINARY" "$REPS" "$SPMV_ITER" "$format"
  fi

done
