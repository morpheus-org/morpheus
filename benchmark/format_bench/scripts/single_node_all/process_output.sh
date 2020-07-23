#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../../.."

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/parser.sh

MACHINE="$1"
COMPILER="$2"
COMP_VERSION="$3"

MACHINE=$(parse_arg $(echo $1 | tr A-Z a-z) "local")
COMPILER=$(parse_arg $(echo $2 | tr A-Z a-z) "gcc")
COMP_VERSION=$(parse_arg $3 "10.1.0")

if $(check_supported_machines $MACHINE); then
  echo "Running on $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/process_output.sh [local|archer|cirrus] [compiler] [version]\n\n"
  echo -e "\t\t local|archer|cirrus: Select for which machine the code is build for."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  exit -1
fi

RESULTS_FILE="$SCRIPT_PATH/../../results/processed_data/single_node_all/$MACHINE"_"$COMPILER"_"$COMP_VERSION.csv"
OUTPUT_PATH="$SCRIPT_PATH/../../results/single_node_all/$MACHINE/$COMPILER/$COMP_VERSION"

mkdir -p $(dirname "$RESULTS_FILE")

# CSV Header
echo "Machine,Matrix,Format,Repetition,Rows,Columns,Nnz,Total,Reader,SpMv,Convert" 2>&1 | tee "$RESULTS_FILE"

for FORMAT_DIR in "$OUTPUT_PATH"/*/
do
  FORMAT=$(basename "$FORMAT_DIR")
  MACHINE_DIR=$(dirname "$FORMAT_DIR")
  for MATRIX_DIR in "$FORMAT_DIR"*
  do
    MATRIX=$(basename "$FORMAT_DIR")
    for REP_DIR in "$MATRIX_DIR"/*
    do
      REP=$(basename "$REP_DIR")
      FILE="$MATRIX_DIR/$REP/output.txt"
      # parse input file
      rows=$(awk '/Matrix Shape/ {printf "%s",$3}' "$FILE")
      columns=$(awk '/Matrix Shape/ {printf "%s",$4}' "$FILE")
      nnz=$(awk '/Matrix Shape/ {printf "%s",$5}' "$FILE")
      total=$(awk '/Total/ {printf "%s",$4}' "$FILE")
      reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")
      convert=$(awk '/Convert/ {printf "%s",$4}' "$FILE")
      spmv=$(awk '/SpMv/ {printf "%s",$4}' "$FILE")

      echo "$MACHINE,$MATRIX,$FORMAT,$REP,$rows,$columns,$nnz,$total,$reader,$spmv,$convert" 2>&1 | tee -a "$RESULTS_FILE"
    done
  done
done