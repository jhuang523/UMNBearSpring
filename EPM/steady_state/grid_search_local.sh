#!/bin/bash -l
MAX_RUNS=${1:-10}
NP=${2:-10}
DATA_DIR=$3


module load impi/2021/5.1
module load conda
source activate modflow
mpirun -np $NP python grid_search_calibration.py --data_output_dir $DATA_DIR --max_runs $MAX_RUNS
