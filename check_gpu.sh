#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --output=gpu_check_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G            # Low memory just for the check

# Optional: Print the node name and GPU details to the output file
echo "Running on node: $(hostname)"
echo "--------------------------"

# The standard command to check GPU health and model
nvidia-smi