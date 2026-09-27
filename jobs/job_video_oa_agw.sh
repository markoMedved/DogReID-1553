#!/bin/bash
#SBATCH --job-name=v_oa_agw
#SBATCH --output=logs_jobs/video_oa_agw_%j.out
#SBATCH --error=logs_jobs/video_oa_agw_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

POOLING="attention"
MODEL_NAME="oa_agw"

echo "=========================================================="
echo "Starting Video-to-Video Training: ${MODEL_NAME} (pooling=${POOLING})"
echo "=========================================================="

python train.py \
    --model ${MODEL_NAME} \
    --world closed \
    --epochs 120 \
    --pooling_type ${POOLING} \
    --val_split 0.2

echo "Video ${MODEL_NAME} job complete!"
