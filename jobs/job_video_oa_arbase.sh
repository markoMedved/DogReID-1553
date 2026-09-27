#!/bin/bash
#SBATCH --job-name=v_oa_arbase
#SBATCH --output=logs_jobs/video_oa_arbase_%j.out
#SBATCH --error=logs_jobs/video_oa_arbase_%j.err
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
MODEL_NAME="oa_arbase"

echo "=========================================================="
echo "Starting Video-to-Video Training: ${MODEL_NAME} (pooling=${POOLING}, full_finetune=True)"
echo "=========================================================="

python train.py \
    --model ${MODEL_NAME} \
    --world closed \
    --epochs 60 \
    --lr 0.0001 \
    --pooling_type ${POOLING} \
    --full_finetune

echo "=========================================================="
echo "Starting Evaluation: Closed-Set Video-to-Video"
echo "=========================================================="

python evaluation/make_csv.py \
    --model_name ${MODEL_NAME} \
    --world_type closed \
    --pooling_type ${POOLING} \
    --full_finetune

echo "=========================================================="
echo "Starting Evaluation: Image-to-Image (Comparison Mode)"
echo "=========================================================="

python evaluation/make_csv.py \
    --model_name ${MODEL_NAME} \
    --world_type closed \
    --pooling_type ${POOLING} \
    --full_finetune \
    --use_images

echo "Video ARBase job complete!"
