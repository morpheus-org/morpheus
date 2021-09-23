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
    EXPERIMENT="spmv-bench"
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

RESULTS_FILE="$ROOT_PATH/core/benchmarks/results/processed/$EXPERIMENT/spmv-$TARGET/$DATASET"_"$MACHINE"_"spmv"_"$FILENAME.csv"
OUTPUT_PATH="$ROOT_PATH/core/benchmarks/results/$EXPERIMENT/spmv-$TARGET/$DATASET"

mkdir -p $(dirname $RESULTS_FILE)

# CSV Header
header="Machine,Matrix,Target,Threads,Reader"
header="$header,SpMv_COO_Custom,SpMv_CSR_Custom,SpMv_DIA_Custom,SpMv_DYN_COO_Custom,SpMv_DYN_CSR_Custom,SpMv_DYN_DIA_Custom"
header="$header,SpMv_COO_Kokkos,SpMv_CSR_Kokkos,SpMv_DIA_Kokkos,SpMv_DYN_COO_Kokkos,SpMv_DYN_CSR_Kokkos,SpMv_DYN_DIA_Kokkos"

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
                SpMv_COO_Custom=$(awk '/SpMv_COO_Custom/ {printf "%s",$4}' "$FILE")
                SpMv_CSR_Custom=$(awk '/SpMv_CSR_Custom/ {printf "%s",$4}' "$FILE")
                SpMv_DIA_Custom=$(awk '/SpMv_DIA_Custom/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_COO_Custom=$(awk '/SpMv_DYN_COO_Custom/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_CSR_Custom=$(awk '/SpMv_DYN_CSR_Custom/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_DIA_Custom=$(awk '/SpMv_DYN_DIA_Custom/ {printf "%s",$4}' "$FILE")

                SpMv_COO_Kokkos=$(awk '/SpMv_COO_Kokkos/ {printf "%s",$4}' "$FILE")
                SpMv_CSR_Kokkos=$(awk '/SpMv_CSR_Kokkos/ {printf "%s",$4}' "$FILE")
                SpMv_DIA_Kokkos=$(awk '/SpMv_DIA_Kokkos/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_COO_Kokkos=$(awk '/SpMv_DYN_COO_Kokkos/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_CSR_Kokkos=$(awk '/SpMv_DYN_CSR_Kokkos/ {printf "%s",$4}' "$FILE")
                SpMv_DYN_DIA_Kokkos=$(awk '/SpMv_DYN_DIA_Kokkos/ {printf "%s",$4}' "$FILE")

                entry="$MACHINE,$MATRIX,$TARGET,$THREAD,$REP,$reader"
                entry="$entry,$SpMv_COO_Custom,$SpMv_CSR_Custom,$SpMv_DIA_Custom"
                entry="$entry,$SpMv_DYN_COO_Custom,$SpMv_DYN_CSR_Custom,$SpMv_DYN_DIA_Custom"
                entry="$entry,$SpMv_COO_Kokkos,$SpMv_CSR_Kokkos,$SpMv_DIA_Kokkos"
                entry="$entry,$SpMv_DYN_COO_Kokkos,$SpMv_DYN_CSR_Kokkos,$SpMv_DYN_DIA_Kokkos"
                
                echo "$entry" 2>&1 | tee -a "$RESULTS_FILE"
            done
        fi
    done
done