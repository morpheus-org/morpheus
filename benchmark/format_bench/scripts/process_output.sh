#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MACHINE="$1"
COMPILER="$2"
COMP_VERSION="$3"

if [ "$COMPILER" == "" ]; then
  COMPILER="gcc"
fi

if [ "$COMP_VERSION" == "" ]; then
  COMP_VERSION="6.3.0"
fi

if [ "$MACHINE" = "local" ] || [ "$MACHINE" = "archer" ] || [ "$MACHINE" = "cirrus" ]; then
  echo "Processing results from $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/process_output.sh [local|archer|cirrus] [compiler] [version]\n\n"
  echo -e "\t\t local|archer|cirrus: Select for which machine the code is build for."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  exit -1
fi

echo "Parameters:"
echo -e "\t MACHINE = $MACHINE"
echo -e "\t COMPILER = $COMPILER"
echo -e "\t VERSION = $COMP_VERSION"

RESULTS_FILE="$SCRIPT_PATH/../results/processed_data/$MACHINE"_"$COMPILER"_"$COMP_VERSION.csv"
OUTPUT_PATH="$SCRIPT_PATH/../results/$MACHINE/$COMPILER/$COMP_VERSION"

mkdir -p $(dirname "$RESULTS_FILE")

# CSV Header
#echo "Machine,Matrix,Format,Repetition,Rows,Columns,Nnz,Total,Reader,Writer,SpMv" 2>&1 | tee "$RESULTS_FILE"
echo "Machine,Matrix,Format,Repetition,Rows,Columns,Nnz,Total,Reader,SpMv,Convert" 2>&1 | tee "$RESULTS_FILE"

for MATRIX_DIR in "$OUTPUT_PATH"/*/
do
  MATRIX=$(basename "$MATRIX_DIR")
  MACHINE_DIR=$(dirname "$MATRIX_DIR")
  for FORMAT_DIR in "$MATRIX_DIR"*
  do
    FORMAT=$(basename "$FORMAT_DIR")
    for REP_DIR in "$FORMAT_DIR"/*
    do
      REP=$(basename "$REP_DIR")
      FILE="$FORMAT_DIR/$REP/output.txt"
      # parse input file
      rows=$(awk '/Matrix Shape/ {printf "%s",$3}' "$FILE")
      columns=$(awk '/Matrix Shape/ {printf "%s",$4}' "$FILE")
      nnz=$(awk '/Matrix Shape/ {printf "%s",$5}' "$FILE")
      total=$(awk '/Total/ {printf "%s",$4}' "$FILE")
      reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")
#      writer=$(awk '/I\/O Write/ {printf "%s",$4}' "$FILE")
      convert=$(awk '/Convert/ {printf "%s",$4}' "$FILE")
      spmv=$(awk '/SpMv/ {printf "%s",$4}' "$FILE")

#      echo "$MACHINE,$MATRIX,$FORMAT,$REP,$rows,$columns,$nnz,$total,$reader,$writer,$spmv" 2>&1 | tee -a "$RESULTS_FILE"
      echo "$MACHINE,$MATRIX,$FORMAT,$REP,$rows,$columns,$nnz,$total,$reader,$spmv,$convert" 2>&1 | tee -a "$RESULTS_FILE"
    done
  done
done