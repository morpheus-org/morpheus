#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MORPHEUS_PATH="$SCRIPT_PATH/../../.."

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/parser.sh

MACHINE="$1"
COMPILER="$2"
COMP_VERSION="$3"

MACHINE=$(parse_arg $(echo $1 | tr A-Z a-z) "local")
COMPILER=$(parse_arg $(echo $2 | tr A-Z a-z) "gcc")
COMP_VERSION=$(parse_arg $3 "7.3.0")

if $(check_supported_machines $MACHINE); then
  echo "Running on $MACHINE"
else
  echo "Invalid input argument."
  echo "Usage:"
  echo -e "\t/path/to/script/parse_stalls.sh [nextgenio] [compiler] [version]\n\n"
  echo -e "\t\t local|archer|cirrus: Select for which machine the code is build for."
  echo -e "\t\t compiler: Compiler used (default is gcc)."
  echo -e "\t\t version: Version of the compiler (default is 6.3.0)."
  exit -1
fi

RESULTS_FILE="$SCRIPT_PATH/../results/processed_data/$MACHINE"_"$COMPILER"_"$COMP_VERSION.csv"
OUTPUT_PATH="$SCRIPT_PATH/../results/$MACHINE/$COMPILER/$COMP_VERSION"

mkdir -p $(dirname "$RESULTS_FILE")

# CSV Header
echo "Machine,Matrix,Format,Repetition,Runtime,Execution Stalls,L1 stalls,L2 Stalls,Memory Stalls,Execution Stall Rate,L1 stall Rate,L2 Stall Rate,Memory Stall Rate" 2>&1 | tee "$RESULTS_FILE"

for FORMAT_DIR in "$OUTPUT_PATH"/*/
do
    FORMAT=$(basename "$FORMAT_DIR")
    MACHINE_DIR=$(dirname "$FORMAT_DIR")
    for MATRIX_DIR in "$FORMAT_DIR"*
    do
        MATRIX=$(basename "$MATRIX_DIR")
        for REP_DIR in "$MATRIX_DIR"/*
        do
            REP=$(basename "$REP_DIR")
            FILE="$MATRIX_DIR/$REP/CYCLE_STALLS.txt"
            # parse input file
            rt=$(awk '/\|           Runtime \(RDTSC\) \[s\]          \|/ {printf "%s",$6}' "$FILE")
            exec_stalls=$(awk '/\|         Total execution stalls         \|/ {printf "%s",$6}' "$FILE")
            L1_stall_perc=$(awk '/\|     Stalls caused by L1D misses \[%\]    \|/ {printf "%s",$9}' "$FILE")
            L2_stall_perc=$(awk '/\|     Stalls caused by L2 misses \[%\]     \|/ {printf "%s",$9}' "$FILE")
            mem_stall_perc=$(awk '/\|    Stalls caused by memory loads \[%\]   \|/ {printf "%s",$9}' "$FILE")
            exec_stall_rate=$(awk '/\|        Execution stall rate \[%\]        \|/ {printf "%s",$7}' "$FILE")
            L1_stall_rate=$(awk '/\|  Stalls caused by L1D misses rate \[%\]  \|/ {printf "%s",$10}' "$FILE")
            L2_stall_rate=$(awk '/\|   Stalls caused by L2 misses rate \[%\]  \|/ {printf "%s",$10}' "$FILE")
            mem_stall_rate=$(awk '/\| Stalls caused by memory loads rate \[%\] \|/ {printf "%s",$10}' "$FILE")

            echo "$MACHINE,$MATRIX,$FORMAT,$REP,$rt,$exec_stalls,$L1_stall_perc,$L2_stall_perc,$mem_stall_perc,$exec_stall_rate,$L1_stall_rate,$L2_stall_rate,$mem_stall_rate" 2>&1 | tee -a "$RESULTS_FILE"
        done
    done
done