#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATRIX_DIR="$SCRIPT_PATH"/../../matrix

#BIN_DIR="$SCRIPT_PATH"/../../examples-build
BIN_DIR="$SCRIPT_PATH"/../../examples-archer-build
BIN=("cusp_spmv" "static_spmv" "dynamic_spmv")

ITER=5

progressfile="$SCRIPT_PATH/progress.txt"

echo "Running spvm benchmark " > "$progressfile"

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
    echo ""$BIN_DIR/$prog" "$MATRIX" $ITER > "$outfile"" >> "$progressfile"
    "$BIN_DIR/$prog" "$MATRIX" $ITER > "$outfile"
  done

done

