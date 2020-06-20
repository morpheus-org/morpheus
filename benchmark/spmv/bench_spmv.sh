#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../matrix

#BIN_DIR="$SCRIPT_PATH"/../../examples-build
BIN_DIR="$SCRIPT_PATH"/../../examples-archer-build
IMPLEMENTATIONS=("cusp" "static" "dynamic")
FORMATS=("coo" "csr")

REPS=5
SPMV_ITER=100

for matdir in "$MATRIX_DIR"/*/
do
  BASE=$(basename $matdir)
  DIR=$(dirname $matdir)
  MATRIX="$DIR/$BASE/$BASE.mtx"

  for impl in "${IMPLEMENTATIONS[@]}"
  do
    for format in "${FORMATS[@]}"
    do
      outdir="$SCRIPT_PATH/results/$BASE/$impl/$format"
      progress="$outdir/progress.txt"

      mkdir -p "$outdir"
      echo "Starting $BASE for $impl implementation and $format format" >&1 | tee "$progress"

      BINARY="$BIN_DIR/$impl"_"$format"_"spmv"
      for rep in `seq -w 1 $REPS`
      do
          fx="$outdir/fx_$rep.txt"
          fy="$outdir/fy_$rep.txt"

          "$BINARY" "$MATRIX" "$outdir" $SPMV_ITER "$rep" >&1 | tee -a "$progress"

      done
    done
  done
done
