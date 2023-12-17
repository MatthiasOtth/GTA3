#!/bin/bash

#SBATCH -A deep_learning
#SBATCH --job=gta3_train
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=/home/maotth/log/%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/maotth/log/%j.err     # where to store error messages

# #SBATCH --mem-per-cpu=4G
# #SBATCH --tmp=8G

#Activate conda
source $HOME/miniconda3/bin/activate gta3
# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
# load module
# use the correct python

mkdir -p $HOME/log

python $HOME/GTA3/zinc_main.py config/zinc/gta3_default.json

# We could copy more results from here to output or any other permanent directory

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

