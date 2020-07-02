#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MACHINE="$1"

if [ "$MACHINE" = "local" ] || [ "$MACHINE" = "archer" ] || [ "$MACHINE" = "cirrus" ]; then
  echo "Processing results from $MACHINE"
else
  echo "Invalid inpug argument."
  echo "Usage:"
  echo -e "\t/path/to/script/process_output.sh [local|archer|cirrus]"
  exit -1
fi

RESULTS_FILE="$SCRIPT_PATH/../results/processed_data_$MACHINE.csv"
OUTPUT_PATH="$SCRIPT_PATH/../results/$MACHINE"

# CSV Header
echo "Machine,Matrix,Version,Repetition,Rows,Columns,Nnz,Total,Reader,Writer,SpMv" 2>&1 | tee "$RESULTS_FILE"

for MATRIX_DIR in "$OUTPUT_PATH"/*/
do
  MATRIX=$(basename "$MATRIX_DIR")
  MACHINE_DIR=$(dirname "$MATRIX_DIR")
  for VERSION_DIR in "$MATRIX_DIR"*
  do
    VERSION=$(basename "$VERSION_DIR")
    for REP_DIR in "$VERSION_DIR"/*
    do
      REP=$(basename "$REP_DIR")
      FILE="$VERSION_DIR/$REP/output.txt"
      # parse input file
      rows=$(awk '/Matrix Shape/ {printf "%s",$3}' "$FILE")
      columns=$(awk '/Matrix Shape/ {printf "%s",$4}' "$FILE")
      nnz=$(awk '/Matrix Shape/ {printf "%s",$5}' "$FILE")
      total=$(awk '/Total/ {printf "%s",$4}' "$FILE")
      reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")
#      writer=$(awk '/I\/O Write/ {printf "%s",$4}' "$FILE")
      spmv=$(awk '/SpMv/ {printf "%s",$4}' "$FILE")

      echo "$MACHINE,$MATRIX,$VERSION,$REP,$rows,$columns,$nnz,$total,$reader,$writer,$spmv" 2>&1 | tee -a "$RESULTS_FILE"
    done
  done
done