#!/bin/sh
# submit.sh
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

CMD="$1"
ROOT_PATH="$2"
OUTPATH="$3"
DATASET="$4"
MACHINE="$5"
TARGET="$6"
MATRIX_PATH="$ROOT_PATH/data/$DATASET"
PROGRESS="$OUTPATH/../progress_$DATASET.txt"
ITER=200
REPS=10
SEED=0

MATRICES=("$MATRIX_PATH/cant"
          "$MATRIX_PATH/consph"
          "$MATRIX_PATH/mac_econ_fwd500"
          "$MATRIX_PATH/mc2depi"
          "$MATRIX_PATH/pdb1HYS"
          "$MATRIX_PATH/pwtk"
          "$MATRIX_PATH/rma10"
          "$MATRIX_PATH/shipsec1")

if [ "$TARGET" == "OpenMP" ];then
    if [ "$MACHINE" == "archer" ]; then
        THREADS=("1" "2" "4" "8" "16" "32" "64")
    elif [ "$MACHINE" == "cirrus" ]; then
        THREADS=("1" "2" "4" "8" "16" "32" "36")
    fi
else
    THREADS=("1")
fi

for thread in "${THREADS[@]}"
do
    export OMP_NUM_THREADS="$thread"
    echo "THREADS=$OMP_NUM_THREADS"
    for iter in {1..$REPS}
    do
        # for each matrix in test space
        for mat in "${MATRICES[@]}"
        do
            BASE=$(basename $mat)
            DIR=$(dirname $mat)
            MATRIX="$DIR/$BASE/$BASE.mtx"

            echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"
            
            OUTDIR="$OUTPATH/$BASE/$iter"
            OUTFILE="$OUTDIR/out.txt"
            mkdir -p $(dirname $OUTFILE)

            $CMD $MATRIX $SEED $ITER 2>&1 | tee -a $OUTFILE
            
        done
    done
done