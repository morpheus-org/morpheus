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
MACHINE="$2"
TARGET="$3"
COMPILER="$4"
OUTPATH="$5"
PROGRESS="$6"
MATRIX="$7"

ITER=200
REPS=5
SEED=0

if [ "$TARGET" == "OpenMP" ];then
    if [ "$MACHINE" == "archer" ]; then
        THREADS=("1" "4" "8" "16" "32" "64")
    elif [ "$MACHINE" == "cirrus" ]; then
        if [ "$COMPILER" == "cuda-11.2" ]; then
            THREADS=("1" "10" "20" "30" "40")
        elif [ "$COMPILER" == "gnu-10.2" ]; then
            THREADS=("1" "4" "8" "16" "32" "36")
        fi
    fi
else
    THREADS=("1")
fi

for thread in "${THREADS[@]}"
do
    echo -e "\tThreads::$thread" 2>&1 | tee -a "$PROGRESS"
    export OMP_NUM_THREADS="$thread"
    for iter in $(seq 1 $REPS)
    do
        echo -e "\t\tRepetition::$iter" 2>&1 | tee -a "$PROGRESS"

        OUTDIR="$OUTPATH/$thread/$iter"
        OUTFILE="$OUTDIR/out.txt"
        mkdir -p $(dirname $OUTFILE)

        $CMD $MATRIX $SEED $ITER 2>&1 | tee -a $OUTFILE    
    done
done