#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. $SCRIPT_PATH/../../../scripts/bash/machine.sh
. $SCRIPT_PATH/../../../scripts/bash/parser.sh

MACHINE=$(parse_arg $(echo $1 | tr A-Z a-z) "local")
COMPILER=$(parse_arg $(echo $2 | tr A-Z a-z) "gcc")
COMP_VERSION=$(parse_arg $3 "10.1.0")
REPS=$(parse_arg $4 "5")
SPMV_ITER=$(parse_arg $5 "100")
TIME=$(parse_arg $6 "00:20:00")

if $(check_supported_machines $MACHINE); then
  echo "Running on $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/run.sh [local|archer|cirrus] [compiler] [version] [reps] [iter] [time]\n\n"
  echo -e "\t\t local|archer|cirrus: Select at which machine the code is running on."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  echo -e "\t\t reps: Number of experiment repetitons (default is 20)."
  echo -e "\t\t iter: Number of repetitions the spMv multiplication is repeated (default is 100)."
  echo -e "\t\t time: Requested time for each run."
  exit -1
fi

BUILD_PATH="$SCRIPT_PATH/../build/$COMPILER/$COMP_VERSION"
RESULTS_PATH="$SCRIPT_PATH/../results/$MACHINE/$COMPILER/$COMP_VERSION"
MATRIX_PATH="$SCRIPT_PATH/../../../matrix"
VERSIONS=("static" "dynamic_01" "dynamic_06" "dynamic_12" "dynamic_20")
FORMAT="0" # COO FORMAT

mkdir -p "$RESULTS_PATH"

for version in "${VERSIONS[@]}"
do
  progress="$RESULTS_PATH/progress"_"$version.txt"
  echo "Starting version $version" 2>&1 | tee "$progress"

  for mat in "$MATRIX_PATH"/*/
  do
    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"
    
    echo -e "\t$BASE" 2>&1 | tee -a "$progress"

    for rep in `seq -w 1 $REPS`
    do
      echo -e "\t\t$rep" 2>&1 | tee -a "$progress"
      outdir="$RESULTS_PATH/$BASE/$version/$rep"
      mkdir -p "$outdir"
      BINARY="$BUILD_PATH/$version"
      NAME="variant_$version"
      FILE="$SCRIPT_PATH/submit.sh"
      FILE_ARGS="$MACHINE $COMPILER $COMP_VERSION $BINARY \
                 $MATRIX $outdir $SPMV_ITER $FORMAT $progress"
      SCHEDULED_JOB=$(configure_scheduler_serial $MACHINE $TIME $NAME $FILE $FILE_ARGS)
      $SCHEDULED_JOB
    done
  done
done
