#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../matrix

#BIN_DIR="$SCRIPT_PATH"/../../examples-build
BIN_DIR="$SCRIPT_PATH"/../../examples-archer-build
IMPLEMENTATIONS=("cusp" "static" "dynamic")
FORMATS=("coo" "csr" "dia" "ell" "hyb")

REPS=5
SPMV_ITER=100

for matdir in "$MATRIX_DIR"/*/
do
  BASE=$(basename $matdir)
  DIR=$(dirname $matdir)
  MATRIX="$DIR/$BASE/$BASE.mtx"

  mkdir -p "$SCRIPT_PATH/results"
  progress="$SCRIPT_PATH/results/progress.txt"
  for impl in "${IMPLEMENTATIONS[@]}"
  do
    echo "Starting $BASE:" >&1 | tee -a "$progress"

    for format in "${FORMATS[@]}"
    do
      echo -e "\t$impl\t$format" >&1 | tee -a "$progress"

      BINARY="$BIN_DIR/$impl"_"$format"_"spmv"
      for rep in `seq -w 1 $REPS`
      do
        echo -e "\t\t\t$rep" >&1 | tee -a "$progress"
        outdir="$SCRIPT_PATH/results/$BASE/$impl/$format/$rep"
        mkdir -p "$outdir"
        "$BINARY" "$MATRIX" "$outdir" $SPMV_ITER >&1 | tee "$outdir/output.txt"

      done
    done
  done
done