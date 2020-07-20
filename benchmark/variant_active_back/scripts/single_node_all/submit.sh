#!/bin/bash --login

MORPHEUS_PATH=$1
MACHINE=$2
COMPILER=$3
COMP_VERSION=$4
BUILD_PATH=$5
MATRIX_PATH=$6
RESULTS_PATH=$7
SPMV_ITER=$8
FORMAT=$9
REPS=${10}

VERSIONS=("static" "dynamic_01" "dynamic_06" "dynamic_12" "dynamic_20")

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/compilers.sh

load_compiler $MORPHEUS_PATH $MACHINE $COMPILER $COMP_VERSION

for version in "${VERSIONS[@]}"
do
    PROGRESS="$RESULTS_PATH/progress"_"$version.txt"
    echo "Starting version $version" 2>&1 | tee "$PROGRESS"

    BINARY="$BUILD_PATH/$version"

    for mat in "$MATRIX_PATH"/*/
    do
        BASE=$(basename $mat)
        DIR=$(dirname $mat)
        MATRIX="$DIR/$BASE/$BASE.mtx"
        
        echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

        for rep in `seq -w 1 $REPS`
        do
            echo -e "\t\t$rep" 2>&1 | tee -a "$PROGRESS"
            OUTDIR="$RESULTS_PATH/$version/$BASE/$rep"
            mkdir -p "$OUTDIR"
            
            LAUNCH_CMD=$(launch_cmd_serial $MORPHEUS_PATH $MACHINE $BINARY)
            LAUNCH_ARGS="$MATRIX $OUTDIR $SPMV_ITER $FORMAT"
        
            $LAUNCH_CMD $LAUNCH_ARGS 2> >(tee -a "$PROGRESS") 1> >(tee "$OUTDIR/output.txt")    
    done
  done

done