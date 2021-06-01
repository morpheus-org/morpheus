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
RESOURCES="--time=$TIME --exclusive --nodes=1 --cpus-per-task=1"
SYSTEM="--partition=standard --qos=standard"
SCHEDULER_ARGS="--account=$ACCOUNT --job-name=morpheus_convert $RESOURCES $SYSTEM"
SCHEDULER_LAUNCER="sbatch"

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically 
#   using threading.
export OMP_NUM_THREADS=1

DATASET="clSpMV"
MATRIX_PATH="$ROOT_PATH/data/$DATASET"
RESULTS_PATH="$ROOT_PATH/benchmarks/spmv/results"

mkdir -p $RESULTS_PATH

PROGRESS="$RESULTS_PATH/progress_$DATASET.txt"

ITER=200
REPS=10
SEED=0
EXECUTABLE="$ROOT_PATH/build/benchmarks/spmv"

MATRICES=("/work/e609/e609/cstyl/morpheus/data/clSpMV/cant"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/consph"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/mac_econ_fwd500"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/mc2depi"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/pdb1HYS"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/pwtk"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/rma10"
        "/work/e609/e609/cstyl/morpheus/data/clSpMV/shipsec1")
# for each matrix in test space
# for mat in "$MATRIX_PATH"/*/
for mat in "${MATRICES[@]}"
do

    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"

    echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"
    
    OUTDIR="$RESULTS_PATH/$DATASET/$BASE"
    OUTFILE="$OUTDIR/out.txt"
    ERRFILE="$OUTDIR/out-err.txt"
    mkdir -p $(dirname $OUTFILE)

    launch_cmd="srun -n 1 --ntasks=1 $EXECUTABLE $MATRIX $SEED $ITER"
    SCHEDULER_FILES="--output=$OUTFILE --error=$ERRFILE"
    $SCHEDULER_LAUNCER $SCHEDULER_ARGS $SCHEDULER_FILES submit.sh "$launch_cmd" "$OUTDIR"
    
done