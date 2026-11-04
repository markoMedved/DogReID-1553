#!/bin/bash
#SBATCH --job-name=open_scl
#SBATCH --output=log_open_results/open_scl_%j.out
#SBATCH --error=log_open_results/open_scl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -x gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p log_open_results checkpoints_open/scl evaluation/csvs_open_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Open-World ECCV-26-SCL (MetaN) with ATTENTION Pooling ---
MODEL="scl"
POOLING="attention"
RUN_NAME="open_scl_attn"
OUTPUT_DIR="checkpoints_open/scl/${RUN_NAME}"
WORLD="open"
BATCH_SIZE=32   # P=8, K=4
K=4
CLIP_LEN=8
EPOCHS=60
LR=0.004
MARGIN=0.7
WEIGHT_DECAY=1e-4
VAL_SPLIT=0
EVAL_PERIOD=5
NUM_WORKERS=8

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Open-World Video Training: ECCV-26-SCL (ATTENTION Pooling)"
echo "  Model        : ${MODEL} (TransReID ViT-B/16 + SCFG [0,4,8] + SCNorm1D + NFC)"
echo "  Pooling      : ${POOLING}"
echo "  Run Name     : ${RUN_NAME}"
echo "  Output Dir   : ${OUTPUT_DIR}"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames @ 256x256"
echo "  Epochs       : ${EPOCHS} (5 warmup epochs, CosineLRScheduler, SGD lr=${LR})"
echo "  Losses       : ID (1.0) + Triplet (margin=${MARGIN}, 1.0) + NFC (3.0, EMA=0.993)"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --pooling_type ${POOLING} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --img_size 256 256 \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --margin ${MARGIN} \
    --weight_decay ${WEIGHT_DECAY} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --num_workers ${NUM_WORKERS} \
    --force_yolo \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Open-World Test-Set Evaluations for SCL (ATTENTION)..."
        echo "=========================================================="
        # 1. Open-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model scl \
            --pooling attention \
            --checkpoint "${CKPT}" \
            --world open \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_open_results/scl_attn_vid2vid" \
            --batch_size 32

        # 2. Open-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model scl \
            --pooling attention \
            --checkpoint "${CKPT}" \
            --world open \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_open_results/scl_attn_img2vid" \
            --batch_size 32
    fi
    echo "Open-World SCL job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
