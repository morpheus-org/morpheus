#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../matrix

BIN_DIR="$SCRIPT_PATH"/../../examples-build
BIN=("cusp_spmv" "static_spmv" "dynamic_spmv")

ITER=5

ACCOUNT="e609"
TIME="walltime=00:20:00"
PLACE="place=scatter:excl"
SELECT="select=1:ncpus=24"
RESOURCES="$SELECT,$TIME,$PLACE"
QSUB="qsub -A $ACCOUNT -l ${RESOURCES}"

for i in "$MATRIX_DIR"/*/
do
  BASE=$(basename $i)
  DIR=$(dirname $i)
  MATRIX="$DIR/$BASE/$BASE.mtx"

  outdir="$SCRIPT_PATH/$BASE"
  mkdir "$outdir"

  for prog in "${BIN[@]}"
  do
    outfile="$outdir/$prog.txt"
    $QSUB -N "$BASE"_"$prog" -v BIN="$BIN_DIR/$prog",MATRIX="$MATRIX",ITER="$ITER",OUTFILE="$outfile"
#    "$BIN_DIR/$prog" "$MATRIX" $ITER > "$outfile"
  done

done