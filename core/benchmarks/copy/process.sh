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
        echo -e "\tFilename::       copy-$FILENAME-$MACHINE-$COMPILER-$DATASET-$TARGET.csv"
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

RESULTS_FILE="$ROOT_PATH/core/benchmarks/results/processed/$EXPERIMENT/copy-$FILENAME-$MACHINE-$COMPILER-$DATASET-$TARGET.csv"
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
