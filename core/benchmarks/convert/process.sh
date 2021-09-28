#!/bin/sh
# process.sh
# 
# EPCC, The University of Edinburgh
# 
# (c) 2021 The University of Edinburgh
# 
# Contributing Authors:
# Christodoulos Stylianou (c.stylianou@ed.ac.uk)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# 	http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# example command-line instruction:
# ./process.sh cirrus gnu-10.2 OpenMP morpheus-bench large_set timings

MACHINE="$1"
COMPILER="$2"
TARGET="$3"
EXPERIMENT="$4"
DATASET="$5"
FILENAME="$6"

if [ "$#" -lt 6 ]; then
    echo "Warning! Only $# out if 6 were provided."
    echo "Arguments Provided are: $1 $2 $3 $4 $5 $6"

    echo "Defaulted arguments:"
    if [ -z "$1" ]; then
        MACHINE="cirrus"
        echo -e "\tMachine::        $MACHINE"
    fi

    if [ -z "$2" ]; then
        COMPILER="gnu-10.2"
        echo -e "\tCompiler::       $COMPILER"
    fi

    if [ -z "$3" ]; then
        TARGET="OpenMP"
        echo -e "\tTarget::          $TARGET"
    fi

    if [ -z "$4" ]; then
        EXPERIMENT="morpheus-bench"
        echo -e "\tExperiment::     $EXPERIMENT"
    fi

    if [ -z "$5" ]; then
        DATASET="large_set"
        echo -e "\tDataset::         $DATASET"
    fi

    if [ -z "$6" ]; then
        FILENAME="timings"
        echo -e "\tFilename::       convert-$FILENAME-$MACHINE-$COMPILER-$DATASET-$TARGET.csv"
    fi
fi

echo -e "\nParsed Runtime Parameters:"
echo -e "=========================="
echo -e "Machine::        $MACHINE"
echo -e "Compiler::       $COMPILER"
echo -e "Target::         $TARGET"
echo -e "Experiment::     $EXPERIMENT"
echo -e "Dataset::        $DATASET"
echo -e "Filename::       $FILENAME"

if [ "$MACHINE" == "archer" ]; then
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
elif [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
fi

echo "Root Path::   $ROOT_PATH"

RESULTS_FILE="$ROOT_PATH/core/benchmarks/results/processed/$EXPERIMENT/convert-$FILENAME-$MACHINE-$COMPILER-$DATASET-$TARGET.csv"
OUTPUT_PATH="$ROOT_PATH/core/benchmarks/results/$EXPERIMENT/convert-$TARGET/$DATASET"

mkdir -p $(dirname $RESULTS_FILE)

# CSV Header
header="Machine,Matrix,Target,Threads,Reader"
header="$header,COO_COO,COO_CSR,COO_DIA,CSR_COO,CSR_CSR,CSR_DIA,DIA_COO,DIA_CSR,DIA_DIA"
header="$header,DYN_COO_COO,DYN_COO_CSR,DYN_COO_DIA,DYN_CSR_COO,DYN_CSR_CSR,DYN_CSR_DIA,DYN_DIA_COO,DYN_DIA_CSR,DYN_DIA_DIA"
header="$header,IN_COO_COO,IN_COO_CSR,IN_COO_DIA,IN_CSR_COO,IN_CSR_CSR,IN_CSR_DIA,IN_DIA_COO,IN_DIA_CSR,IN_DIA_DIA"

echo "$header"  2>&1 | tee "$RESULTS_FILE"

for MATRIX_DIR in "$OUTPUT_PATH"/*
do
    MATRIX=$(basename "$MATRIX_DIR")
    for THREADS_DIR in "$MATRIX_DIR"/*
    do
        if [[ -d $THREADS_DIR ]]; then
            THREAD=$(basename "$THREADS_DIR")
            for REPS_DIR in "$THREADS_DIR"/*
            do
                REP=$(basename "$REPS_DIR")
                FILE="$MATRIX_DIR/$THREAD/$REP/out.txt"

                # parse input file
                reader=$(awk '/I\/O Read/ {printf "%s",$4}' "$FILE")

                COO_COO=$(awk '/Convert_COO_COO/ {printf "%s",$4}' "$FILE")
                COO_CSR=$(awk '/Convert_COO_CSR/ {printf "%s",$4}' "$FILE")
                COO_DIA=$(awk '/Convert_COO_DIA/ {printf "%s",$4}' "$FILE")
                CSR_COO=$(awk '/Convert_CSR_COO/ {printf "%s",$4}' "$FILE")
                CSR_CSR=$(awk '/Convert_CSR_CSR/ {printf "%s",$4}' "$FILE")
                CSR_DIA=$(awk '/Convert_CSR_DIA/ {printf "%s",$4}' "$FILE")
                DIA_COO=$(awk '/Convert_DIA_COO/ {printf "%s",$4}' "$FILE")
                DIA_CSR=$(awk '/Convert_DIA_CSR/ {printf "%s",$4}' "$FILE")
                DIA_DIA=$(awk '/Convert_DIA_DIA/ {printf "%s",$4}' "$FILE")

                DYN_COO_COO=$(awk '/Convert_DYN_COO_COO/ {printf "%s",$4}' "$FILE")
                DYN_COO_CSR=$(awk '/Convert_DYN_COO_CSR/ {printf "%s",$4}' "$FILE")
                DYN_COO_DIA=$(awk '/Convert_DYN_COO_DIA/ {printf "%s",$4}' "$FILE")
                DYN_CSR_COO=$(awk '/Convert_DYN_CSR_COO/ {printf "%s",$4}' "$FILE")
                DYN_CSR_CSR=$(awk '/Convert_DYN_CSR_CSR/ {printf "%s",$4}' "$FILE")
                DYN_CSR_DIA=$(awk '/Convert_DYN_CSR_DIA/ {printf "%s",$4}' "$FILE")
                DYN_DIA_COO=$(awk '/Convert_DYN_DIA_COO/ {printf "%s",$4}' "$FILE")
                DYN_DIA_CSR=$(awk '/Convert_DYN_DIA_CSR/ {printf "%s",$4}' "$FILE")
                DYN_DIA_DIA=$(awk '/Convert_DYN_DIA_DIA/ {printf "%s",$4}' "$FILE")

                IN_COO_COO=$(awk '/Convert_IN_COO_COO/ {printf "%s",$4}' "$FILE")
                IN_COO_CSR=$(awk '/Convert_IN_COO_CSR/ {printf "%s",$4}' "$FILE")
                IN_COO_DIA=$(awk '/Convert_IN_COO_DIA/ {printf "%s",$4}' "$FILE")
                IN_CSR_COO=$(awk '/Convert_IN_CSR_COO/ {printf "%s",$4}' "$FILE")
                IN_CSR_CSR=$(awk '/Convert_IN_CSR_CSR/ {printf "%s",$4}' "$FILE")
                IN_CSR_DIA=$(awk '/Convert_IN_CSR_DIA/ {printf "%s",$4}' "$FILE")
                IN_DIA_COO=$(awk '/Convert_IN_DIA_COO/ {printf "%s",$4}' "$FILE")
                IN_DIA_CSR=$(awk '/Convert_IN_DIA_CSR/ {printf "%s",$4}' "$FILE")
                IN_DIA_DIA=$(awk '/Convert_IN_DIA_DIA/ {printf "%s",$4}' "$FILE")

                entry="$MACHINE,$MATRIX,$TARGET,$THREAD,$reader"
                entry="$entry,$COO_COO,$COO_CSR,$COO_DIA,$CSR_COO,$CSR_CSR,$CSR_DIA,$DIA_COO,$DIA_CSR,$DIA_DIA"
                entry="$entry,$DYN_COO_COO,$DYN_COO_CSR,$DYN_COO_DIA,$DYN_CSR_COO,$DYN_CSR_CSR,$DYN_CSR_DIA,$DYN_DIA_COO,$DYN_DIA_CSR,$DYN_DIA_DIA"
                entry="$entry,$IN_COO_COO,$IN_COO_CSR,$IN_COO_DIA,$IN_CSR_COO,$IN_CSR_CSR,$IN_CSR_DIA,$IN_DIA_COO,$IN_DIA_CSR,$IN_DIA_DIA"
                
                echo "$entry" 2>&1 | tee -a "$RESULTS_FILE"
            done
        fi
    done
done
