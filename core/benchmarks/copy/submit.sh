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
OUTPATH="$2"
PROGRESS="$3"
MATRIX="$4"

REPS=1
ITER=10

THREADS=("1")

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

        $CMD $MATRIX $ITER 2>&1 | tee -a $OUTFILE    
    done
done