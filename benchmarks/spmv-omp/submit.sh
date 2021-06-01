#!/bin/sh

CMD="$1"
OUTPATH="$2"

# THREADS=("1" "2" "4" "8" "16" "32" "64" "128")
THREADS=("32")

for iter in {1..10}
do
    for THREAD in "${THREADS[@]}"
    do
        export OMP_NUM_THREADS="$THREAD"
        export OMP_PLACES=cores
        
        OUTFILE="$OUTPATH/$iter/$THREAD/out.txt"
        mkdir -p $(dirname $OUTFILE)

        $CMD 2>&1 | tee -a $OUTFILE
    done
done

