#SBATCH --job-name=ok_conduit_network_test
#SBATCH --time=144:00:00
#SBATCH --ntasks=2
#SBATCH --mem=500gb
#SBATCH --tmp=200g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huan1428@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=1
#SBATCH --output=output/job_%j.out
#SBATCH --error=output/job_%j.err

module load impi/2021/5.1
module load conda
source activate openkarst_2.0
mpirun -np $SLURM_NTASKS python  ../../../src/scripts/openkarst/mpi_openkarst.py --sim_list_file sim_list.txt --verbose #put in task here 