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
echo "Machine,Matrix,Format,Repetition,Runtime,Power,Power Ram,Flops,Flops AVX,Read Bandwidth,Write Bandwidth,Read Volume,Write Volume,Bandwidth,Memory,Arithmetic Intensity" 2>&1 | tee "$RESULTS_FILE"

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
            FILE="$MATRIX_DIR/$REP/MEM_DP.txt"
            # parse input file
            rt=$(awk '/\|        Runtime \(RDTSC\) \[s\]        \|/ {printf "%s",$6}' "$FILE")
            power=$(awk '/\|             Power \[W\]             \|/ {printf "%s",$5}' "$FILE")
            power_ram=$(awk '/\|           Power DRAM \[W\]          \|/ {printf "%s",$6}' "$FILE")
            flops=$(awk '/\|            DP \[MFLOP\/s\]           \|/ {printf "%s",$5}' "$FILE")
            flops_avx=$(awk '/\|          AVX DP \[MFLOP\/s\]         \|/ {printf "%s",$6}' "$FILE")
            read_bw=$(awk '/\|  Memory read bandwidth \[MBytes\/s\] \|/ {printf "%s",$7}' "$FILE")
            write_bw=$(awk '/\| Memory write bandwidth \[MBytes\/s\] \|/ {printf "%s",$7}' "$FILE")
            read_mem=$(awk '/\|  Memory read data volume \[GBytes\] \|/ {printf "%s",$8}' "$FILE")
            write_mem=$(awk '/\| Memory write data volume \[GBytes\] \|/ {printf "%s",$8}' "$FILE")
            bw=$(awk '/\|    Memory bandwidth \[MBytes\/s\]    \|/ {printf "%s",$6}' "$FILE")
            mem=$(awk '/\|    Memory data volume \[GBytes\]    \|/ {printf "%s",$7}' "$FILE")
            ar_intensity=$(awk '/\|       Operational intensity       \|/ {printf "%s",$5}' "$FILE")

            echo "$MACHINE,$MATRIX,$FORMAT,$REP,$rt,$power,$power_ram,$flops,$flops_avx,$read_bw,$write_bw,$read_mem,$write_mem,$bw,$mem,$ar_intensity" 2>&1 | tee -a "$RESULTS_FILE"
        done
    done
done