#!/bin/bash
#SBATCH --job-name=v_dino_bot
#SBATCH --output=logs_jobs/video_dinov2_bot_%j.out
#SBATCH --error=logs_jobs/video_dinov2_bot_%j.err
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

echo "=========================================================="
echo "Starting Video-to-Video Training: DINOv2 + BoT (pooling=${POOLING}, unfreeze_blocks=2)"
echo "=========================================================="

python train.py \
    --model dinov2 \
    --reid_method bot \
    --world closed \
    --epochs 51 \
    --lr 2e-05 \
    --clip_len 8 \
    --pooling_type ${POOLING} \
    --no_full_finetune \
    --unfreeze_blocks 2

echo "=========================================================="
echo "Starting Evaluation: Closed-Set Video-to-Video"
echo "=========================================================="

python evaluation/make_csv.py \
    --model_name bot \
    --backbone dinov2 \
    --world_type closed \
    --pooling_type ${POOLING}

echo "=========================================================="
echo "Starting Evaluation: Image-to-Image (Comparison Mode)"
echo "=========================================================="

python evaluation/make_csv.py \
    --model_name bot \
    --backbone dinov2 \
    --world_type closed \
    --pooling_type ${POOLING} \
    --use_images

echo "DINOv2 BoT job complete!"
