#!/bin/bash
#SBATCH --job-name=vmd_def
#SBATCH --output=logs_scl/val_md_default_%j.out
#SBATCH --error=logs_scl/val_md_default_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -x gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p logs_scl checkpoints/megadescriptor_val

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment: MegaDescriptor Val 0 - Default (Full Unfrozen, BNNeck, Triplet+CE, LS, RE, Cosine, AdamW) ---
MODEL="megadescriptor"
POOLING="attention"
RUN_NAME="val_md_default_full_cos"
OUTPUT_DIR="checkpoints/megadescriptor_val/${RUN_NAME}"
WORLD="closed"
BATCH_SIZE=32
K=4
CLIP_LEN=8
CHUNK_SIZE=16
ACCUM_STEPS=2
EPOCHS=51
LR=2e-05
BACKBONE_LR_FACTOR=0.1
MARGIN=0.3
WEIGHT_DECAY=1e-05
OPTIMIZER="adamw"
LR_SCHED="cosine"
LR_DELAY_EPOCHS=25
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
VAL_SPLIT=0.2
EVAL_PERIOD=5
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting MegaDescriptor 20% Val Tuning: DEFAULT"
echo "  Run Name        : ${RUN_NAME}"
echo "  Backbone        : Fully unfrozen (100% trainable)"
echo "  Neck / Loss     : BNNeck, CrossEntropy + BatchHard Triplet"
echo "  Regularization  : Label Smoothing 0.1, Random Erasing 0.5"
echo "  Optimizer / LR  : AdamW (1e-5), Cosine delay=25 (Head 2e-5, Backbone 2e-6)"
echo "  Validation Split: 20% (val_split=0.2, eval_period=5)"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --pooling_type ${POOLING} \
    --full_finetune \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --chunk_size ${CHUNK_SIZE} \
    --accum_steps ${ACCUM_STEPS} \
    --img_size 224 224 \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --weight_decay ${WEIGHT_DECAY} \
    --optimizer ${OPTIMIZER} \
    --lr_sched ${LR_SCHED} \
    --lr_delay_epochs ${LR_DELAY_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --warmup_factor ${WARMUP_FACTOR} \
    --margin ${MARGIN} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --num_workers ${NUM_WORKERS} \
    --force_yolo \
    --resume

echo "Job complete!"
