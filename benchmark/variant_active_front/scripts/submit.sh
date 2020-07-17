#!/bin/bash --login

__PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. $__PATH/../../../scripts/bash/machine.sh
. $__PATH/../../../scripts/bash/compilers.sh

MACHINE=$1
COMPILER=$2
COMP_VERSION=$3
BINARY=$4
MATRIX=$5
OUTDIR=$6
SPMV_ITER=$7
FORMAT=$8
PROGRESS=$9

load_compiler $MACHINE $COMPILER $COMP_VERSION
LAUNCH_CMD=$(launch_cmd_serial $MACHINE $BINARY)
LAUNCH_ARGS="$MATRIX $OUTDIR $SPMV_ITER $FORMAT"

$LAUNCH_CMD $LAUNCH_ARGS 2> >(tee -a "$PROGRESS") 1> >(tee "$OUTDIR/output.txt")