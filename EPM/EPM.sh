#!/bin/bash -l
#SBATCH --job-name =EPM_test
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --mem=500gb
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=souce015@umn.edu
#SBATCH -p pkkang
#SBATCH --nodes=1

cd ~/BearSpring/BearSpringUMN/EPM
module load anaconda/miniconda3_4.8.3-jupyter
source activate modflow6_HPC
jupyter nbconvert --to notebook --execute EPM_ws_simple_transient.ipynb --output EPM_ws_transient_out.ipynb --ExecutePreprocessor.timeout=43200
