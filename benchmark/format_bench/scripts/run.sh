#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MORPHEUS_PATH="$SCRIPT_PATH/../../../.."

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/parser.sh

MACHINE=$(parse_arg $(echo $1 | tr A-Z a-z) "local")
COMPILER=$(parse_arg $(echo $2 | tr A-Z a-z) "gcc")
COMP_VERSION=$(parse_arg $3 "10.1.0")
REPS=$(parse_arg $4 "5")
SPMV_ITER=$(parse_arg $5 "100")
TIME=$(parse_arg $6 "00:20:00")
QUEUE=$(parse_arg $7 "standard")

if $(check_supported_machines $MACHINE); then
  echo "Running on $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/run.sh [local|archer|cirrus] [compiler] [version] [reps] [iter] [time] [queue]\n\n"
  echo -e "\t\t local|archer|cirrus: Select at which machine the code is running on."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  echo -e "\t\t reps: Number of experiment repetitons (default is 20)."
  echo -e "\t\t iter: Number of repetitions the spMv multiplication is repeated (default is 100)."
  echo -e "\t\t time: Requested time for each run."
  echo -e "\t\t queue: Requested queue to run."
  exit -1
fi

BUILD_PATH="$SCRIPT_PATH/../build/$COMPILER/$COMP_VERSION"
RESULTS_PATH="$SCRIPT_PATH/../results/$MACHINE/$COMPILER/$COMP_VERSION"
MATRIX_PATH="$SCRIPT_PATH/../../../matrix/format_bench"

#FORMATS=("coo", "csr", "dia", "ell", "hyb")
FORMATS=("coo" "csr" "hyb") # for now these do not require fill in

mkdir -p "$RESULTS_PATH"

for FORMAT in "${FORMATS[@]}"
do
  PROGRESS="$RESULTS_PATH/progress"_"$FORMAT.txt"
  echo "Starting format $FORMAT" 2>&1 | tee "$PROGRESS"

  BINARY="$BUILD_PATH/dynamic_selection"
  NAME="format_$FORMAT"
  OUTDIR="$RESULTS_PATH/$FORMAT"

  FILE="$SCRIPT_PATH/submit.sh"
  FILE_ARGS="$MORPHEUS_PATH $MACHINE $COMPILER $COMP_VERSION \
              $BINARY $MATRIX_PATH $OUTDIR $SPMV_ITER $FORMAT $REPS \
              $PROGRESS"
  
  SCHEDULED_JOB=$(configure_scheduler_serial $MORPHEUS_PATH $MACHINE $QUEUE $TIME $NAME $FILE $FILE_ARGS)
  $SCHEDULED_JOB
done
