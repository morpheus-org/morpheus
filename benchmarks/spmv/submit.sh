#!/bin/sh

CMD="$1"
OUTPATH="$2"

for iter in {1..50}
do

OUTFILE="$OUTPATH/$iter/out.txt"
mkdir -p $(dirname $OUTFILE)

$CMD 2>&1 | tee -a $OUTFILE

done