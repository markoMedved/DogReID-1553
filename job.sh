#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs_swin/train_%j.out
#SBATCH --error=logs_swin/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Load your conda function
source $(conda info --base)/etc/profile.d/conda.sh

# Go to your code directory
cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

# Activate the env
conda activate project

echo "Running on $(hostname) with $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Finally launch
python train.py --model vit --world closed --lr 5e-05 --margin 0.5