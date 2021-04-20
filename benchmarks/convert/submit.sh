#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=morpheus_convert
#SBATCH --time=3:00:0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=e609           
#SBATCH --partition=standard
#SBATCH --qos=standard

# Setup the job environment (this module needs to be loaded before any other modules)
module load epcc-job-env

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically 
#   using threading.
export OMP_NUM_THREADS=1

# SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# ROOT_PATH="${SCRIPT_PATH}/../.."
ROOT_PATH="/work/e609/e609/cstyl/morpheus"

DATASET="clSpMV"
MATRIX_PATH="$ROOT_PATH/data/$DATASET"
RESULTS_PATH="$ROOT_PATH/benchmarks/convert/results"

mkdir -p $RESULTS_PATH

PROGRESS="$RESULTS_PATH/progress_$DATASET.txt"

# REPS=10
REPS=2
EXECUTABLE="$ROOT_PATH/build/benchmarks/convert"

# for each matrix in test space
for mat in "$MATRIX_PATH"/*/
do

    BASE=$(basename $mat)
    DIR=$(dirname $mat)
    MATRIX="$DIR/$BASE/$BASE.mtx"

    echo -e "\t$BASE" 2>&1 | tee -a "$PROGRESS"

    OUTFILE="$RESULTS_PATH/$DATASET/$BASE/out.txt"
    mkdir -p $(dirname $OUTFILE)
    echo "srun -n 1 --ntasks=1 $EXECUTABLE $MATRIX $REPS" 2>&1 | tee -a "$PROGRESS"
    echo "srun -n 1 --ntasks=1 $EXECUTABLE $MATRIX $REPS" 2>&1 | tee -a "$OUTFILE"
    srun -n 1 --ntasks=1 $EXECUTABLE $MATRIX $REPS 2>&1 | tee -a "$OUTFILE"

done