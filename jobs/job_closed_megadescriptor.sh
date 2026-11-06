#!/bin/bash
#SBATCH --job-name=cl_md
#SBATCH --output=log_baseline_results/closed_megadescriptor_%j.out
#SBATCH --error=log_baseline_results/closed_megadescriptor_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH -x gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

mkdir -p log_baseline_results checkpoints/megadescriptor evaluation/csvs_closed_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Closed-World MegaDescriptor (Swin-B/L) Optimal Baseline (50 Epochs) ---
MODEL="megadescriptor"
REID_METHOD="bot"
RUN_NAME="closed_megadescriptor_ep50"
OUTPUT_DIR="checkpoints/megadescriptor/${RUN_NAME}"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=32
K=4
CLIP_LEN=8
CHUNK_SIZE=16
EPOCHS=50
LR=2e-05
WEIGHT_DECAY=1e-05
OPTIMIZER="adamw"
BACKBONE_LR_FACTOR=0.1
LR_SCHED="cosine"
LR_DELAY_EPOCHS=25
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
LABEL_SMOOTHING=0.0
RE_PROB=0.0
MARGIN=0.3
TRIPLET_TYPE="hard"
VAL_SPLIT=0.0
EVAL_PERIOD=50
ACCUM_STEPS=2
NUM_WORKERS=8

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Closed-World Training: MegaDescriptor Optimal Model"
echo "  Model           : ${MODEL} (Swin-Large, FULL FINE-TUNE)"
echo "  Pooling         : ${POOLING}"
echo "  Run Name        : ${RUN_NAME}"
echo "  Output Dir      : ${OUTPUT_DIR}"
echo "  World           : ${WORLD}"
echo "  Batch / K       : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length     : ${CLIP_LEN} frames @ 224x224 (chunk_size=${CHUNK_SIZE})"
echo "  Epochs          : ${EPOCHS} (warmup: ${WARMUP_EPOCHS} ep, sched: ${LR_SCHED}, delay: ${LR_DELAY_EPOCHS})"
echo "  Regularization  : No Random Erasing (0.0), No Label Smoothing (0.0)"
echo "  Losses          : Hard CE + Batch-Hard Triplet (margin ${MARGIN})"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --reid_method ${REID_METHOD} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --pooling_type ${POOLING} \
    --full_finetune \
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
    --label_smoothing ${LABEL_SMOOTHING} \
    --re_prob ${RE_PROB} \
    --margin ${MARGIN} \
    --triplet_type ${TRIPLET_TYPE} \
    --use_bnneck \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --num_workers ${NUM_WORKERS} \
    --force_yolo \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Closed-World Test-Set Evaluations for MegaDescriptor..."
        echo "=========================================================="
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model ${MODEL} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_closed_results/megadescriptor_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model ${MODEL} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_closed_results/megadescriptor_img2vid" \
            --batch_size 32
    fi
    echo "Closed-World MegaDescriptor job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
