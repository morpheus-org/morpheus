#!/bin/sh

MACHINE="$1"
TIME="$2"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Invalid arguments.. Exiting.."
    exit -1
fi

if [ "$MACHINE" == "archer" ]; then
    # Setup the job environment (this module needs to be loaded before any other modules)
    module load epcc-job-env
    ROOT_PATH="/work/e609/e609/cstyl/morpheus"
fi

if [ "$MACHINE" == "cirrus" ]; then
    ROOT_PATH="/lustre/home/e609/cstyl/morpheus"
fi

ACCOUNT="e609"
RESOURCES="--time=$TIME --exclusive --nodes=1 --tasks-per-node=1 --cpus-per-task=128"
SYSTEM="--partition=standard --qos=standard"
SCHEDULER_ARGS="--account=$ACCOUNT --job-name=morpheus_omp $RESOURCES $SYSTEM"
SCHEDULER_LAUNCER="sbatch"

DATASET="clSpMV"
MATRIX_PATH="$ROOT_PATH/data/$DATASET"
RESULTS_PATH="$ROOT_PATH/benchmarks/spmv-omp/results"

mkdir -p $RESULTS_PATH

PROGRESS="$RESULTS_PATH/progress_$DATASET.txt"

REPS=200
# REPS=5
SEED=0
EXECUTABLE="$ROOT_PATH/build/benchmarks/spmv-omp"

THREADS=("1" "2" "4" "8" "16" "32" "64")
# THREADS=("1" "2" "4" "8" "16" "32")
# THREADS=("64" "128")

MATRICES=(
    # "/work/e609/e609/cstyl/morpheus/data/clSpMV/cant"
    #     "/work/e609/e609/cstyl/morpheus/data/clSpMV/consph"
    #     "/work/e609/e609/cstyl/morpheus/data/clSpMV/mac_econ_fwd500"
    #     "/work/e609/e609/cstyl/morpheus/data/clSpMV/mc2depi"
    #     "/work/e609/e609/cstyl/morpheus/data/clSpMV/pdb1HYS"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/pwtk"
        # "/work/e609/e609/cstyl/morpheus/data/clSpMV/rma10"
        # "/work/e609/e609/cstyl/morpheus/data/clSpMV/shipsec1"
        )
# for each matrix in test space
# for mat in "$MATRIX_PATH"/*/
for mat in "${MATRICES[@]}"
do

    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"

    echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

    OUTFILE="$RESULTS_PATH/$DATASET/$BASE/out.txt"
    ERRFILE="$RESULTS_PATH/$DATASET/$BASE/out-err.txt"
    
    OUTPATH="$RESULTS_PATH/$DATASET/$BASE"
    mkdir -p $OUTPATH

    launch_cmd="srun -n 1 --hint=nomultithread --ntasks=1 $EXECUTABLE $MATRIX $SEED $REPS"
    SCHEDULER_FILES="--output=$OUTFILE --error=$ERRFILE"
    $SCHEDULER_LAUNCER $SCHEDULER_ARGS $SCHEDULER_FILES submit.sh "$launch_cmd" "$OUTPATH"

done