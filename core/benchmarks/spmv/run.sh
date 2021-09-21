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

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Invalid arguments.. Exiting.."
    exit -1
fi

if [ "$MACHINE" == "archer" ]; then
    # Setup the job environment (this module needs to be loaded before any other modules)
    module load epcc-job-env
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
    TARGETS=("Serial" "OpenMP")
elif [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
    TARGETS=("Serial" "OpenMP" "Cuda")
    if [ "$COMPILER" != "cuda-11.2" ] || [ "$COMPILER" == "gnu-10.2" ]; then
         echo "Invalid compiler argument ($COMPILER).. Exiting.."
        exit -1
    fi
fi

for target in "${TARGETS[@]}"
do
if [ "$MACHINE" == "archer" ]; then
    MAX_CPUS="--cpus-per-task=64"
    SYSTEM="--partition=standard --qos=standard"
elif [ "$MACHINE" == "cirrus" ]; then
    MAX_CPUS="--cpus-per-task=36"
    if [ "$target" == "Cuda" ];then
        SYSTEM="--gres=gpu:4 --partition=gpu-cascade --qos=gpu"
    else
        SYSTEM="--partition=standard --qos=standard"
    fi
fi

    ACCOUNT="e609"
    RESOURCES="--time=$TIME --exclusive --nodes=1 $MAX_CPUS"
    SCHEDULER_ARGS="--account=$ACCOUNT --job-name=spmv_$target $RESOURCES $SYSTEM"
    SCHEDULER_LAUNCER="sbatch"

    DATASET="clSpMV"
    RESULTS_PATH="$ROOT_PATH/core/benchmarks/results/spmv-$target"
    EXECUTABLE="$ROOT_PATH/build-$COMPILER-release/core/benchmarks/MorpheusCore_Benchmarks_Spmv_$target"

    mkdir -p $RESULTS_PATH

    OUTDIR="$RESULTS_PATH/$DATASET"
    OUTFILE="$OUTDIR/out.txt"
    ERRFILE="$OUTDIR/out-err.txt"
    mkdir -p $(dirname $OUTFILE)

    SUBMISSION_SCRIPT="$ROOT_PATH/core/benchmarks/spmv/submit.sh"
    launch_cmd="srun -n 1 --hint=nomultithread --ntasks=1 $EXECUTABLE"
    SCHEDULER_FILES="--output=$OUTFILE --error=$ERRFILE"
    $SCHEDULER_LAUNCER $SCHEDULER_ARGS $SCHEDULER_FILES $SUBMISSION_SCRIPT "$launch_cmd" "$ROOT_PATH" "$OUTDIR" "$DATASET" "$MACHINE" "$target"
done
