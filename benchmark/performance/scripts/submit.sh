#!/bin/bash --login

MORPHEUS_PATH=$1
MACHINE=$2
COMPILER=$3
COMP_VERSION=$4
BINARY=$5
MATRIX_PATH=$6
REF_OUTDIR=$7
FORMAT=$(echo $8 | tr A-Z a-z)
REPS=$9
PROGRESS=${10}

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/compilers.sh

load_compiler $MORPHEUS_PATH $MACHINE $COMPILER $COMP_VERSION

module load likwid
PERFORMANCE_GROUPS=("MEM_DP" "CYCLE_STALLS")

for mat in "$MATRIX_PATH"/*/
  do
    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"
    
    echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

    for group in "${PERFORMANCE_GROUPS[@]}"
    do
        for rep in `seq -w 1 $REPS`
        do
            echo -e "\t\t$rep" 2>&1 | tee -a "$PROGRESS"
            OUTDIR="$REF_OUTDIR/$BASE/$rep"
            mkdir -p "$OUTDIR"
            
            LIKWID="likwid-perfctr -C 0 -g "$group" -m -o $OUTDIR/$group.txt"

            LAUNCH_CMD="srun -n 1 --ntasks=1"
            LAUNCH_ARGS="$MATRIX"
            $LAUNCH_CMD $LIKWID $BINARY $LAUNCH_ARGS 
        done
    done
  done


