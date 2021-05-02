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

REPS=10
# REPS=5
SEED=0
EXECUTABLE="$ROOT_PATH/build/benchmarks/spmv"

# for each matrix in test space
for mat in "$MATRIX_PATH"/*/
do

    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"

    echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

    OUTFILE="$RESULTS_PATH/$DATASET/$BASE/out.txt"
    ERRFILE="$RESULTS_PATH/$DATASET/$BASE/out-err.txt"
    mkdir -p $(dirname $OUTFILE)

    launch_cmd="srun -n 1 --ntasks=1 $EXECUTABLE $MATRIX $SEED $REPS 2>&1 | tee -a $OUTFILE"
    SCHEDULER_FILES="--output=$OUTFILE --error=$ERRFILE"
    $SCHEDULER_LAUNCER $SCHEDULER_ARGS $SCHEDULER_FILES submit.sh "$launch_cmd"

done