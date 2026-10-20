#!/bin/bash
#SBATCH --job-name=val_full
#SBATCH --output=log_baseline/val_full_finetune_%j.out
#SBATCH --error=log_baseline/val_full_finetune_%j.err
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

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

mkdir -p log_baseline checkpoints_baseline

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Full Fine-Tuning (100% of Parameters Trainable) ---
MODEL="baseline"
REID_METHOD="bot"
RUN_NAME="val_lr2e-5_ep51_ms25-35_full_finetune"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=32
K=4
CLIP_LEN=8
CHUNK_SIZE=16
EPOCHS=51
LR=2e-05
WEIGHT_DECAY=1e-05
OPTIMIZER="adamw"
BACKBONE_LR_FACTOR=0.1
LR_SCHED="multistep"
LR_MILESTONES="25,35"
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
VAL_SPLIT=0.2
EVAL_PERIOD=5
ACCUM_STEPS=2
NUM_WORKERS=8

echo "=========================================================="
echo "Starting Video-to-Video Validation Ablation: Full Fine-Tuning"
echo "  Model           : ${MODEL} (VideoReID / reid_method=${REID_METHOD})"
echo "  Run Name        : ${RUN_NAME}"
echo "  Backbone        : DINOv2 ViT-B/14 (100% trainable, full fine-tune)"
echo "  Val Split       : ${VAL_SPLIT} (held-out validation identities)"
echo "  Optimizer       : ${OPTIMIZER} (weight_decay=${WEIGHT_DECAY})"
echo "  Learning Rate   : Head ${LR}, Backbone $(python -c "print(${LR} * ${BACKBONE_LR_FACTOR})")"
echo "  Epochs          : ${EPOCHS} (WarmupMultiStepLR milestones=${LR_MILESTONES})"
echo "=========================================================="

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --reid_method ${REID_METHOD} \
    --run_name ${RUN_NAME} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --chunk_size ${CHUNK_SIZE} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --optimizer ${OPTIMIZER} \
    --lr_sched ${LR_SCHED} \
    --lr_milestones ${LR_MILESTONES} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --warmup_factor ${WARMUP_FACTOR} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --full_finetune \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --force_yolo

echo "Full Fine-Tuning Validation job complete!"
