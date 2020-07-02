#!/bin/bash

module load gcc/8.2.0
module load cmake/3.17.3

export OMP_NUM_THREADS=1

SCRIPT_PATH="$1"
MATRIX_DIR="$2"
RESULTS_PATH="$3"
PROGRESS_FILE="$4"
BINARY="$5"
REPS="$6"
SPMV_ITER="$7"
FORMAT="$8"

if [ "$FORMAT" == "coo" ];then
  fmt_int=0
elif [ "$FORMAT" == "csr" ]; then
  fmt_int=1
elif [ "$FORMAT" == "dia" ]; then
  fmt_int=2
elif [ "$FORMAT" == "ell" ]; then
  fmt_int=3
elif [ "$FORMAT" == "hyb" ]; then
  fmt_int=4
elif [ "$FORMAT" == "dense" ]; then
  fmt_int=5
fi

for matdir in "$MATRIX_DIR"/*/
do
  BASE=$(basename $matdir)
  DIR=$(dirname $matdir)
  MATRIX="$DIR/$BASE/$BASE.mtx"

  echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS_FILE"

  for rep in `seq -w 1 $REPS`
    do
      echo -e "\t\t$rep" 2>&1 | tee -a "$PROGRESS_FILE"
      outdir="$RESULTS_PATH/$BASE/$VERSION/$rep"
      mkdir -p "$outdir"
      "$BINARY" "$MATRIX" "$outdir" "$SPMV_ITER" "$fmt_int" 2> >(tee -a "$PROGRESS_FILE") 1> >(tee "$outdir/output.txt")
    done

done


