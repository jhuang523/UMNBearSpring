#!/bin/bash -l

timestamp=$(date +%Y%m%d_%H%M%S)

#SBATCH --job-name=grid_search_calibration
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --mem=500gb
#SBATCH --tmp=200g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huan1428@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=2
#SBATCH --output=logs/calibration_%j.out
#SBATCH --error=logs/calibration_%j.err

export SLURM_OUTPUT="logs/calibration_${timestamp}.out"
export SLURM_ERROR="logs/calibration_${timestamp}.err"

module load impi/2021/5.1
module load conda
source activate modflow
mpirun -np $SLURM_NTASKS python grid_search_calibration.py --data_output_dir EPM_2layer_unconfined --max_runs 1000 > "$SLURM_OUTPUT" 2> "$SLURM_ERROR"
