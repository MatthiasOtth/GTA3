#!/bin/bash

#SBATCH -A deep_learning
#SBATCH --job=gta3
#SBATCH --gpus=1  # max
#SBATCH --cpus-per-task=2  # max=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=8G
#SBATCH --output=/home/%u/GTA3/logs/euler/%j.out
#SBATCH --error=/home/%u/GTA3/logs/euler/%j.err

# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# ensure logs dirs exit
mkdir -p $HOME/GTA3/logs/euler

# run
cd $HOME/GTA3

source .venv/bin/activate
echo "python: $(which python3)"
echo "python version: $(python3 --version)"

srun python3 cluster_main.py config/cluster/gta3_500k.json --seed 0
srun python3 cluster_main.py config/cluster/gta3_500k.json --seed 1
srun python3 cluster_main.py config/cluster/gta3_500k.json --seed 2
srun python3 cluster_main.py config/cluster/gta3_500k.json --seed 3

cd --

# We could copy more results from here to output or any other permanent directory

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

