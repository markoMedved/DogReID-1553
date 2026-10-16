#!/bin/bash
#SBATCH --job-name=v_dino_agw_v1
#SBATCH --output=logs_dinov2/video_dinov2_agw_v1_%j.out
#SBATCH --error=logs_dinov2/video_dinov2_agw_v1_%j.err
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
export PYTHONPATH=OpenAnimals:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (DINOv2 AGW with Pure [CLS] Representation + Weighted Triplet) ---
MODEL="dinov2_agw"
RUN_NAME="v1_cls_lr3.5e-4"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=64   # P×K = 64 (P=16, K=4)
K=4
CLIP_LEN=8
EPOCHS=50
LR=3.5e-04
BACKBONE_LR_FACTOR=0.1
UNFREEZE_BLOCKS=2
VAL_SPLIT=0          # train on full train split, evaluate on test split
EVAL_PERIOD=5
ACCUM_STEPS=1
NUM_WORKERS=12

echo "=========================================================="
echo "Starting Video-to-Video Training: DINOv2 AGW"
echo "  Model        : ${MODEL}"
echo "  Run Name     : ${RUN_NAME}"
echo "  Representation: Pure [CLS] token (Native ViT anchor)"
echo "  Pooling      : Temporal Attention (${POOLING})"
echo "  Loss         : Weighted Regularized Soft-margin Triplet (0.0) + CE"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames"
echo "  Resolution   : 224x224 (DINOv2 patch size 14x14)"
echo "  Learning Rate: Head ${LR}, Backbone $(python -c "print(${LR} * ${BACKBONE_LR_FACTOR})")"
echo "  Epochs       : ${EPOCHS}"
echo "  Scheduler    : WarmupMultiStepLR (milestones 25, 35)"
echo "  Unfrozen Blk : top ${UNFREEZE_BLOCKS} blocks (blocks 0..9 frozen)"
echo "  Workers      : ${NUM_WORKERS}"
echo "=========================================================="

python train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --unfreeze_blocks ${UNFREEZE_BLOCKS} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS}

echo "Video DINOv2 AGW job complete!"
