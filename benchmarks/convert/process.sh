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
BENCHMARK="convert"
OUTPUT_PATH="$ROOT_PATH/benchmarks/$BENCHMARK/results/$DATASET"
RESULTS_FILE="$ROOT_PATH/benchmarks/$BENCHMARK/results/$DATASET"_"$MACHINE"_"$BENCHMARK"_"$FILENAME.csv"

# CSV Header
header="Machine,Matrix,Reader,Convert_COO_COO,Convert_COO_CSR,Convert_COO_DIA"
header="$header,Convert_CSR_COO,Convert_CSR_CSR,Convert_CSR_DIA,Convert_DIA_COO"
header="$header,Convert_DIA_CSR,Convert_DIA_DIA,Convert_DYN_COO_COO"
header="$header,Convert_DYN_COO_CSR,Convert_DYN_COO_DIA,Convert_DYN_CSR_COO"
header="$header,Convert_DYN_CSR_CSR,Convert_DYN_CSR_DIA,Convert_DYN_DIA_COO"
header="$header,Convert_DYN_DIA_CSR,Convert_DYN_DIA_DIA"

echo "$header"  2>&1 | tee "$RESULTS_FILE"

for MATRIX_DIR in "$OUTPUT_PATH"/*/
do
    MATRIX=$(basename "$MATRIX_DIR")
    FILE="$MATRIX_DIR/out.txt"
    
    # parse input file
    reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")
    convert_coo_coo=$(awk '/Convert_COO_COO/ {printf "%s",$4}' "$FILE")
    convert_coo_csr=$(awk '/Convert_COO_CSR/ {printf "%s",$4}' "$FILE")
    convert_coo_dia=$(awk '/Convert_COO_DIA/ {printf "%s",$4}' "$FILE")
    convert_csr_coo=$(awk '/Convert_CSR_COO/ {printf "%s",$4}' "$FILE")
    convert_csr_csr=$(awk '/Convert_CSR_CSR/ {printf "%s",$4}' "$FILE")
    convert_csr_dia=$(awk '/Convert_CSR_DIA/ {printf "%s",$4}' "$FILE")
    convert_dia_coo=$(awk '/Convert_DIA_COO/ {printf "%s",$4}' "$FILE")
    convert_dia_csr=$(awk '/Convert_DIA_CSR/ {printf "%s",$4}' "$FILE")
    convert_dia_dia=$(awk '/Convert_DIA_DIA/ {printf "%s",$4}' "$FILE")
    convert_dyn_coo_coo=$(awk '/Convert_DYN_COO_COO/ {printf "%s",$4}' "$FILE")
    convert_dyn_coo_csr=$(awk '/Convert_DYN_COO_CSR/ {printf "%s",$4}' "$FILE")
    convert_dyn_coo_dia=$(awk '/Convert_DYN_COO_DIA/ {printf "%s",$4}' "$FILE")
    convert_dyn_csr_coo=$(awk '/Convert_DYN_CSR_COO/ {printf "%s",$4}' "$FILE")
    convert_dyn_csr_csr=$(awk '/Convert_DYN_CSR_CSR/ {printf "%s",$4}' "$FILE")
    convert_dyn_csr_dia=$(awk '/Convert_DYN_CSR_DIA/ {printf "%s",$4}' "$FILE")
    convert_dyn_dia_coo=$(awk '/Convert_DYN_DIA_COO/ {printf "%s",$4}' "$FILE")
    convert_dyn_dia_csr=$(awk '/Convert_DYN_DIA_CSR/ {printf "%s",$4}' "$FILE")
    convert_dyn_dia_dia=$(awk '/Convert_DYN_DIA_DIA/ {printf "%s",$4}' "$FILE")

    entry="$MACHINE,$MATRIX,$reader,$convert_coo_coo,$convert_coo_csr,$convert_coo_dia"
    entry="$entry,$convert_csr_coo,$convert_csr_csr,$convert_csr_dia,$convert_dia_coo"
    entry="$entry,$convert_dia_csr,$convert_dia_dia,$convert_dyn_coo_coo,$convert_dyn_coo_csr"
    entry="$entry,$convert_dyn_coo_dia,$convert_dyn_csr_coo,$convert_dyn_csr_csr,$convert_dyn_csr_dia"
    entry="$entry,$convert_dyn_dia_coo,$convert_dyn_dia_csr,$convert_dyn_dia_dia"
    
    echo "$entry" 2>&1 | tee -a "$RESULTS_FILE"
done 
