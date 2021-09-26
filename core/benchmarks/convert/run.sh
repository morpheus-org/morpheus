#!/bin/sh
# run.sh
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
TIME="$3"
EXPERIMENT="$4"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Invalid arguments.. Exiting.."
    exit -1
fi

if [ -z "$4" ]; then
    EXPERIMENT="convert-bench"
fi

echo "Machine::     $MACHINE"
echo "Compiler::    $COMPILER"
echo "Time::        $TIME"
echo "Experiment::  $EXPERIMENT"

if [ "$MACHINE" == "archer" ]; then
    # Setup the job environment (this module needs to be loaded before any other modules)
    module load epcc-job-env
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
elif [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
fi

TARGETS=("Serial")

echo "Root Path::   $ROOT_PATH"
echo "Targets:: ${TARGETS[*]}"

for target in "${TARGETS[@]}"
do
    if [ "$MACHINE" == "archer" ]; then
        MAX_CPUS="--cpus-per-task=64"
        SYSTEM="--partition=standard --qos=standard"
    elif [ "$MACHINE" == "cirrus" ]; then
        if [ "$COMPILER" == "cuda-11.2" ]; then
            MAX_CPUS="--cpus-per-task=40"
            SYSTEM="--gres=gpu:4 --partition=gpu-cascade --qos=gpu"
        elif [ "$COMPILER" == "gnu-10.2" ]; then
            MAX_CPUS="--cpus-per-task=36"
            SYSTEM="--partition=standard --qos=standard"
        else
            echo "Invalid compiler argument ($COMPILER).. Exiting.."
            exit -1
        fi
    fi

    ACCOUNT="e609"
    RESOURCES="--time=$TIME --exclusive --nodes=1 $MAX_CPUS"
    SCHEDULER_ARGS="--account=$ACCOUNT --job-name=convert_$target $RESOURCES $SYSTEM"
    SCHEDULER_LAUNCER="sbatch"

    DATASET="large_set"
    RESULTS_PATH="$ROOT_PATH/core/benchmarks/results/$EXPERIMENT/convert-$target"
    EXECUTABLE="$ROOT_PATH/build-$COMPILER-release/core/benchmarks/MorpheusCore_Benchmarks_Convert"
    MATRIX_PATH="$ROOT_PATH/data/$DATASET"

    mkdir -p $RESULTS_PATH

    SUBMISSION_SCRIPT="$ROOT_PATH/core/benchmarks/convert/submit.sh"
    launch_cmd="srun -n 1 --hint=nomultithread --ntasks=1 $EXECUTABLE"

    # for each matrix in test space
    for mat in $MATRIX_PATH/*
    do
        if [[ -d $mat ]]
        then
    	    BASE=$(basename $mat)
            DIR=$(dirname $mat)
            MATRIX="$DIR/$BASE/$BASE.mtx"

            OUTDIR="$RESULTS_PATH/$DATASET/$BASE"
            OUTFILE="$OUTDIR/out.txt"
            ERRFILE="$OUTDIR/out-err.txt"
            mkdir -p $(dirname $OUTFILE)

            PROGRESS="$RESULTS_PATH/progress_${DATASET}_${BASE}.txt"
            echo -e "Matrix::$BASE" 2>&1 | tee -a "$PROGRESS"
            SCHEDULER_FILES="--output=$OUTFILE --error=$ERRFILE"

   	        $SCHEDULER_LAUNCER $SCHEDULER_ARGS $SCHEDULER_FILES $SUBMISSION_SCRIPT "$launch_cmd" "$OUTDIR" "$PROGRESS" "$MATRIX" "$COMPILER"
	    fi
    done
done
