#!/bin/bash
#SBATCH --job-name=v_sbs
#SBATCH --output=logs_jobs/video_sbs_%j.out
#SBATCH --error=logs_jobs/video_sbs_%j.err
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

# --- Experiment Settings (SBS in DogReID Framework) ---
MODEL="sbs"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=64   # P×K = 64 (P=16, K=4)
K=4
CLIP_LEN=8
EPOCHS=120
LR=3.5e-04
VAL_SPLIT=0.2
ACCUM_STEPS=1

echo "=========================================================="
echo "Starting Video-to-Video Training: SBS (Native Framework)"
echo "  Model        : ${MODEL}"
echo "  Method       : bot"
echo "  Pooling      : ${POOLING}"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames"
echo "  Learning Rate: ${LR}"
echo "  Epochs       : ${EPOCHS}"
echo "=========================================================="

python train.py \
    --model ${MODEL} \
    --reid_method bot \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --val_split ${VAL_SPLIT} \
    --accum_steps ${ACCUM_STEPS} \
    --full_finetune

echo "Video SBS job complete!"
