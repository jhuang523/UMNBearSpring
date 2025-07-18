#!/bin/bash -l
#SBATCH --job-name=grid_search_calibration
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --mem=500gb
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huan1428@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=10
#SBATCH --output=calibration.out

module loadimpi/2021/5.1
mpirun -np $SLURM_NTASKS python grid_search_calibration.py 