#!/bin/bash
#SBATCH --job-name=v_tfclip_v1
#SBATCH --output=logs_jobs/video_tfclip_v1_%j.out
#SBATCH --error=logs_jobs/video_tfclip_v1_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p logs_jobs checkpoints_tfclip

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (TF-CLIP: Text-Free CLIP with TMD + SSP) ---
MODEL="tfclip"
RUN_NAME="v1_tfclip_vitb16"
WORLD="closed"
BATCH_SIZE=16   # P×K = 16 (P=4, K=4, matching base TF-CLIP paper)
K=4
CLIP_LEN=8
EPOCHS=60
LR=5e-06        # Base LR = 5e-6 (bias=2x, classifier_proj_temp=10x, matching TF-CLIP Stage 2)
BACKBONE_LR_FACTOR=1.0
WEIGHT_DECAY=1e-04
VAL_SPLIT=0     # train on full train split, evaluate on query/gallery test set
EVAL_PERIOD=5
ACCUM_STEPS=1
NUM_WORKERS=12

echo "=========================================================="
echo "Starting Video-to-Video Training: TF-CLIP (AAAI 2024)"
echo "  Model        : ${MODEL}"
echo "  Run Name     : ${RUN_NAME}"
echo "  Backbone     : OpenAI CLIP ViT-B/16"
echo "  Components   : CLIP-Memory (Centroids) + SSP (Prompt) + TMD (Diffusion)"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames"
echo "  Resolution   : 224x224"
echo "  Learning Rate: Base ${LR} (bias=2x, classifier_proj_temp=10x)"
echo "  Epochs       : ${EPOCHS} (10-epoch linear warmup)"
echo "  Scheduler    : WarmupMultiStepLR (milestones 30, 50)"
echo "  Workers      : ${NUM_WORKERS}"
echo "=========================================================="

python train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --weight_decay ${WEIGHT_DECAY} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --full_finetune

echo "Video TF-CLIP job complete!"
