#!/bin/bash
#SBATCH --job-name=dino_masking
#SBATCH --output=logs_jobs/dino_masking_%j.out
#SBATCH --error=logs_jobs/dino_masking_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=gwn04

set -e

source $(conda info --base)/etc/profile.d/conda.sh
cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
conda activate project

mkdir -p logs_jobs

echo "=================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Host Node     : $(hostname)"
echo "Assigned GPU  : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'No GPU found')"
echo "Starting Background Masking DINOv2 Training (100 epochs)..."
echo "=================================================="

python train.py --model dinov2 --world closed --mask_dog --epochs 100

echo "Training completed. Running Background Masking Evaluation..."
python evaluation/make_csv.py --model_name dinov2 --world_type closed --mask_dog --use_images
