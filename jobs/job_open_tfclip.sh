#!/bin/bash
#SBATCH --job-name=open_tfclip
#SBATCH --output=log_open_results/open_tfclip_%j.out
#SBATCH --error=log_open_results/open_tfclip_%j.err
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
mkdir -p log_open_results checkpoints_open_results evaluation/csvs_open_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (Open-World TF-CLIP: Text-Free CLIP with TMD + SSP) ---
MODEL="tfclip"
RUN_NAME="open_v1_tfclip_vitb16"
OUTPUT_DIR="checkpoints_open_results/tfclip/${RUN_NAME}"
WORLD="open"
BATCH_SIZE=16   # P×K = 16 (P=4, K=4, matching base TF-CLIP paper)
K=4
CLIP_LEN=8
EPOCHS=60
LR=5e-06        # Base LR = 5e-6 (bias=2x, classifier_proj_temp=10x, matching TF-CLIP Stage 2)
BACKBONE_LR_FACTOR=1.0
WEIGHT_DECAY=1e-04
VAL_SPLIT=0
EVAL_PERIOD=5
ACCUM_STEPS=1
NUM_WORKERS=12

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Open-World Video Training: TF-CLIP (AAAI 2024)"
echo "  Model        : ${MODEL}"
echo "  Run Name     : ${RUN_NAME}"
echo "  Output Dir   : ${OUTPUT_DIR}"
echo "  World        : ${WORLD}"
echo "  Batch / K    : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length  : ${CLIP_LEN} frames"
echo "  Resolution   : 224x224"
echo "  Learning Rate: Base ${LR} (bias=2x, classifier_proj_temp=10x)"
echo "  Epochs       : ${EPOCHS} (10-epoch linear warmup, multistep 30, 50)"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
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
    --full_finetune \
    --force_yolo \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Open-World Test-Set Evaluations for TF-CLIP..."
        echo "=========================================================="
        # 1. Open-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model tfclip \
            --checkpoint "${CKPT}" \
            --world open \
            --output_dir "evaluation/csvs_open_results/tfclip_vid2vid" \
            --batch_size 32

        # 2. Open-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model tfclip \
            --checkpoint "${CKPT}" \
            --world open \
            --query_images \
            --output_dir "evaluation/csvs_open_results/tfclip_img2vid" \
            --batch_size 32
    fi
    echo "Open-World TF-CLIP job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
