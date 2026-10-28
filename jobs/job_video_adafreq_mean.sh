#!/bin/bash
#SBATCH --job-name=v_af_mean
#SBATCH --output=logs_scl/video_adafreq_mean_%j.out
#SBATCH --error=logs_scl/video_adafreq_mean_%j.err
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
mkdir -p logs_scl checkpoints/adafreq/adafreq_mean_ep150_bs32 evaluation/csvs_closed_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Closed-World ECCV-24 AdaFreq with MEAN Pooling ---
MODEL="adafreq"
POOLING="mean"
RUN_NAME="adafreq_mean_ep150_bs32"
OUTPUT_DIR="checkpoints/adafreq/${RUN_NAME}"
WORLD="closed"
BATCH_SIZE=32   # P=8, K=4 (exact paper setup)
K=4
CLIP_LEN=8
EPOCHS=150      # Exact paper setup: 150 epochs
LR=0.001        # Exact paper setup: SGD lr=0.001 with Cosine decay
WEIGHT_DECAY=1e-4
VAL_SPLIT=0
EVAL_PERIOD=5
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Closed-World Video Training: ECCV-24 AdaFreq (MEAN Pooling)"
echo "  Model        : ${MODEL} (TransReID ViT-B/16 stride=[12,12], mu=0.5, lambda=0.1)"
echo "  Pooling      : ${POOLING}"
echo "  Run Name     : ${RUN_NAME}"
echo "  Output Dir   : ${OUTPUT_DIR}"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames @ 256x256"
echo "  Augmentation : Rot(15 deg), Brightness(p=0.5), Contrast(p=0.5), Pad(10)"
echo "  Epochs       : ${EPOCHS} (5 warmup epochs, CosineLRScheduler, SGD lr=${LR})"
echo "  Losses       : ID (0.9*CE_orig + 0.1*CE_high) + Triplet (0.7*SoftTri_orig + 0.2*SoftTri_high + 0.1*SmoothL1)"
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
        echo "Running Closed-World Test-Set Evaluations for AdaFreq (MEAN)..."
        echo "=========================================================="
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model adafreq \
            --pooling mean \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_closed_results/adafreq_mean_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model adafreq \
            --pooling mean \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_closed_results/adafreq_mean_img2vid" \
            --batch_size 32
    fi
    echo "Closed-World ECCV-24 AdaFreq (MEAN) job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
