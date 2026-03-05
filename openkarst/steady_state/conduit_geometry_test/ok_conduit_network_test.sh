#!/bin/bash -l        
#SBATCH --job-name=ok_conduit_network_test
#SBATCH --time=144:00:00
#SBATCH --ntasks=10
#SBATCH --mem=500gb
#SBATCH --tmp=200g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huan1428@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=2
#SBATCH --output=output/logs/job_%j.out
#SBATCH --error=output/logs/job_%j.err

module load impi/2021/5.1
module load conda
source activate openkarst
mpirun -np $SLURM_NTASKS python  ../../../src/scripts/openkarst/mpi_openkarst.py --sim_list_file sim_list.txt --log_path "simulation_$(date +%Y%m%d_%H%M%S).log"
)#put in task here 