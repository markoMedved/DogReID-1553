#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs_dinov2/train_%j.out
#SBATCH --error=logs_dinov2/train_%j.err
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

python train.py \
    --model swin \
    --world closed \
    --clip_len 32 \
    --batch_size 64 \
    --k 4 \
    --lr 3e-05 \
    --margin 0.3 \
    --weight_decay 0.01