#!/bin/sh

MACHINE="$1"
FILENAME="$2"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Invalid arguments.. Exiting.."
    exit -1
fi

if [ "$MACHINE" == "archer" ]; then
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
fi

if [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
fi

DATASET="clSpMV"
BENCHMARK="spmv-omp"
OUTPUT_PATH="$ROOT_PATH/benchmarks/$BENCHMARK/results/$DATASET"
RESULTS_FILE="$ROOT_PATH/benchmarks/$BENCHMARK/results/$DATASET"_"$MACHINE"_"$BENCHMARK"_"$FILENAME.csv"

THREADS=("1" "2" "4" "8" "16" "32" "64" "128")

# CSV Header
header="Machine,Matrix,Threads,Reader,Set_Vecs,SpMv_COO,SpMv_CSR,SpMv_DIA,SpMv_DYN_COO,SpMv_DYN_CSR,SpMv_DYN_DIA"

echo "$header"  2>&1 | tee "$RESULTS_FILE"

for MATRIX_DIR in "$OUTPUT_PATH"/*
do
    MATRIX=$(basename "$MATRIX_DIR")
    for THREAD in "${THREADS[@]}"
    do
        FILE="$MATRIX_DIR/$THREAD/out.txt"
        
        # parse input file
        reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")
        Set_Vecs=$(awk '/Set_Vecs/ {printf "%s",$4}' "$FILE")
        SpMv_COO=$(awk '/SpMv_COO/ {printf "%s",$4}' "$FILE")
        SpMv_CSR=$(awk '/SpMv_CSR/ {printf "%s",$4}' "$FILE")
        SpMv_DIA=$(awk '/SpMv_DIA/ {printf "%s",$4}' "$FILE")
        SpMv_DYN_COO=$(awk '/SpMv_DYN_COO/ {printf "%s",$4}' "$FILE")
        SpMv_DYN_CSR=$(awk '/SpMv_DYN_CSR/ {printf "%s",$4}' "$FILE")
        SpMv_DYN_DIA=$(awk '/SpMv_DYN_DIA/ {printf "%s",$4}' "$FILE")

        entry="$MACHINE,$MATRIX,$THREAD,$reader,$Set_Vecs,$SpMv_COO,$SpMv_CSR,$SpMv_DIA,$SpMv_DYN_COO,$SpMv_DYN_CSR,$SpMv_DYN_DIA" 
        echo "$entry" 2>&1 | tee -a "$RESULTS_FILE"
    done
done 
