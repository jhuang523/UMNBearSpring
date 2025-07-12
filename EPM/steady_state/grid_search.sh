#!/bin/bash -l

#SBATCH --job-name=grid_search_calibration
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --mem=500gb
#SBATCH --tmp=200g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huan1428@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=2
#SBATCH --output=logs/calibration_%j.out
#SBATCH --error=logs/calibration_%j.err

module load impi/2021/5.1
module load conda
source activate modflow
mpirun -np $SLURM_NTASKS python grid_search_calibration.py --data_output_dir EPM_2layer_unconfined --max_runs 1000
