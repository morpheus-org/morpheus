#!/bin/bash --login

MORPHEUS_PATH=$1
MACHINE=$2
COMPILER=$3
COMP_VERSION=$4
BUILD_PATH=$5
MATRIX_PATH=$6
RESULTS_PATH=$7
SPMV_ITER=$8
REPS=$9

FORMATS=("coo" "csr" "dia" "ell" "hyb")
#FORMATS=("coo" "csr" "hyb") # for now these do not require fill in

. $MORPHEUS_PATH/scripts/bash/machine.sh
. $MORPHEUS_PATH/scripts/bash/compilers.sh

load_compiler $MORPHEUS_PATH $MACHINE $COMPILER $COMP_VERSION

for FORMAT in "${FORMATS[@]}"
do
    PROGRESS="$RESULTS_PATH/progress"_"$FORMAT.txt"
    echo "Starting format $FORMAT" 2>&1 | tee "$PROGRESS"

    if [ "$FORMAT" == "coo" ];then fmt=0
    elif [ "$FORMAT" == "csr" ]; then fmt=1
    elif [ "$FORMAT" == "dia" ]; then fmt=2
    elif [ "$FORMAT" == "ell" ]; then fmt=3
    elif [ "$FORMAT" == "hyb" ]; then fmt=4
    elif [ "$FORMAT" == "dense" ]; then fmt=5
    fi

    BINARY="$BUILD_PATH/dynamic_selection"

    for mat in "$MATRIX_PATH"/*/
    do
        BASE=$(basename $mat)
        DIR=$(dirname $mat)
        MATRIX="$DIR/$BASE/$BASE.mtx"

        echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

        for rep in `seq -w 1 $REPS`
        do
            echo -e "\t\t$rep" 2>&1 | tee -a "$PROGRESS"
            OUTDIR="$RESULTS_PATH/$FORMAT/$BASE/$rep"
            mkdir -p "$OUTDIR"
            
            LAUNCH_CMD=$(launch_cmd_serial $MORPHEUS_PATH $MACHINE $BINARY)
            LAUNCH_ARGS="$MATRIX $OUTDIR $SPMV_ITER $fmt"

            $LAUNCH_CMD $LAUNCH_ARGS 2> >(tee -a "$PROGRESS") 1> >(tee "$OUTDIR/output.txt")    
    done
  done
done
