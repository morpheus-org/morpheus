#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

MACHINE="$1"

if [ "$MACHINE" = "local" ] || [ "$MACHINE" = "archer" ] || [ "$MACHINE" = "cirrus" ]; then
  echo "Checking correctness in $MACHINE"
else
  echo "Invalid inpug argument."
  echo "Usage:"
  echo -e "\t/path/to/script/process_output.sh [local|archer|cirrus]"
  exit -1
fi

LOGGER="$SCRIPT_PATH/../results/correctness_$MACHINE.txt"
OUTPUT_PATH="$SCRIPT_PATH/../results/$MACHINE"

echo "Checking for each matrix and each version, repetitions yield to the same output." 2>&1 | tee "$LOGGER"

# loop over each matrix
# loop over each version
# check repetition outputs same within the version
ctr_x=0
ctr_y=0
for MATRIX_DIR in "$OUTPUT_PATH"/*/
do
  for VERSION_DIR in "$MATRIX_DIR"/*
  do
    REF_X="$VERSION_DIR/1/fx.txt"
    REF_Y="$VERSION_DIR/1/fy.txt"

    for REP_DIR in "$VERSION_DIR"/*
    do
      TEST_X="$REP_DIR/fx.txt"
      TEST_Y="$REP_DIR/fy.txt"
      DIFF_X=$(diff "$REF_X" "$TEST_X")
      DIFF_Y=$(diff "$REF_Y" "$TEST_Y")
      if [ "$DIFF_X" != "" ];then
          echo "Test Failed:$TEST_X differ from $REF_X" >> "$LOGGER"
          ctr_x=$((ctr_x + 1))
      fi

      if [ "$DIFF_Y" != "" ];then
          echo "Test Failed:$TEST_Y differ from $REF_Y" >> "$LOGGER"
          ctr_y=$((ctr_y + 1))
      fi
    done
  done
done

if [ $ctr_x -eq 0 ] && [ $ctr_y -eq 0 ];then
  echo "All tests PASSED." 2>&1 | tee -a "$LOGGER"
fi

echo "Checking for each matrix output is the same across versions." 2>&1 | tee -a "$LOGGER"

## loop over each matrix
## for each version
## compare output from rep 1 between versions
ctr_x=0
ctr_y=0
for MATRIX_DIR in "$OUTPUT_PATH"/*/
do
  MATRIX=$(basename "$MATRIX_DIR")
  REF_X="$MATRIX_DIR/cusp/1/fx.txt"
  REF_Y="$MATRIX_DIR/cusp/1/fy.txt"
  ctr_x=0
  ctr_y=0
  ctr_tests=0
  for VERSION_DIR in "$MATRIX_DIR"*
  do
    VERSION=$(basename "$VERSION_DIR")
    TEST_X="$VERSION_DIR/1/fx.txt"
    TEST_Y="$VERSION_DIR/1/fy.txt"

    DIFF_X=$(diff "$REF_X" "$TEST_X")
    DIFF_Y=$(diff "$REF_Y" "$TEST_Y")

    if [ "$DIFF_X" != "" ];then
        echo "Test Failed:$TEST_X differ from $REF_X" >> "$LOGGER"
        ctr_x=$((ctr_x + 1))
    fi

    if [ "$DIFF_Y" != "" ];then
        echo "Test Failed:$TEST_Y differ from $REF_Y" >> "$LOGGER"
        ctr_y=$((ctr_y + 1))
    fi
  done
done

if [ $ctr_x -eq 0 ] && [ $ctr_y -eq 0 ];then
  echo "All tests PASSED." 2>&1 | tee -a "$LOGGER"
fi