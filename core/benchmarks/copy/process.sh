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

MACHINE="$1"
COMPILER="$2"
TARGET="$3"
EXPERIMENT="$4"
FILENAME="$5"
DATASET="$6"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Invalid arguments.. Exiting.."
    exit -1
fi

if [ -z "$3" ]; then
    echo "Missing target.. Exiting.."
    exit -1
fi

if [ -z "$4" ]; then
    echo "Warning! Experiment, filename and dataset not provided. Using default options."
    EXPERIMENT="copy-bench"
    FILENAME="timings"
    DATASET="clSpMV"
fi

echo "Machine::     $MACHINE"
echo "Compiler::    $COMPILER"
echo "Target::      $TARGET"
echo "Experiment::  $EXPERIMENT"
echo "Filename::    $FILENAME"
echo "Dataset::     $DATASET"


if [ "$MACHINE" == "archer" ]; then
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
elif [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
fi

echo "Root Path::   $ROOT_PATH"

RESULTS_FILE="$ROOT_PATH/core/benchmarks/results/processed/$EXPERIMENT/copy-$TARGET/$DATASET"_"$MACHINE"_"copy"_"$FILENAME.csv"
OUTPUT_PATH="$ROOT_PATH/core/benchmarks/results/$EXPERIMENT/copy-$TARGET/$DATASET"

mkdir -p $(dirname $RESULTS_FILE)

# CSV Header
header="Machine,Matrix,Target,Threads,Reader"
header="$header,COO_Deep,CSR_Deep,DIA_Deep"
header="$header,COO_Elem,CSR_Elem,DIA_Elem"

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

                COO_Deep=$(awk '/Copy_COO_Deep/ {printf "%s",$4}' "$FILE")
                CSR_Deep=$(awk '/Copy_CSR_Deep/ {printf "%s",$4}' "$FILE")
                DIA_Deep=$(awk '/Copy_DIA_Deep/ {printf "%s",$4}' "$FILE")

                COO_Elem=$(awk '/Copy_COO_Elem/ {printf "%s",$4}' "$FILE")
                CSR_Elem=$(awk '/Copy_CSR_Elem/ {printf "%s",$4}' "$FILE")
                DIA_Elem=$(awk '/Copy_DIA_Elem/ {printf "%s",$4}' "$FILE")

                entry="$MACHINE,$MATRIX,$TARGET,$THREAD,$reader"
                entry="$entry,$COO_Deep,$CSR_Deep,$DIA_Deep"
                entry="$entry,$COO_Elem,$CSR_Elem,$DIA_Elem"
                
                echo "$entry" 2>&1 | tee -a "$RESULTS_FILE"
            done
        fi
    done
done
