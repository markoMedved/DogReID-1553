#!/bin/bash
#SBATCH --job-name=v_md_f_at
#SBATCH --output=logs_scl/video_megadescriptor_full_attn_%j.out
#SBATCH --error=logs_scl/video_megadescriptor_full_attn_%j.err
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
mkdir -p logs_scl checkpoints/megadescriptor evaluation/csvs_closed_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: MegaDescriptor-B-224 FULL FINE-TUNE (DINOv2 Baseline Settings) ---
MODEL="megadescriptor"
POOLING="attention"
RUN_NAME="megadescriptor_b_full_finetune_${POOLING}_ep51_cos_d25"
OUTPUT_DIR="checkpoints/megadescriptor/${RUN_NAME}"
WORLD="closed"
BATCH_SIZE=32   # P=8, K=4
K=4
CLIP_LEN=8
CHUNK_SIZE=16
ACCUM_STEPS=2
EPOCHS=51
LR=2e-05        # Head LR (Backbone gets 0.1x = 2e-06)
BACKBONE_LR_FACTOR=0.1
MARGIN=0.3
WEIGHT_DECAY=1e-05
OPTIMIZER="adamw"
LR_SCHED="cosine"
LR_DELAY_EPOCHS=25
LR_ETA_MIN=7e-08
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
VAL_SPLIT=0.0
EVAL_PERIOD=5
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting MegaDescriptor-B-224 Video Training: FULL FINE-TUNE"
echo "  Backbone        : ${MODEL} (BVRA/MegaDescriptor-B-224 Swin-B 1024-d, 100% unfrozen)"
echo "  Temporal Pooling: ${POOLING}"
echo "  Run Name        : ${RUN_NAME}"
echo "  Output Dir      : ${OUTPUT_DIR}"
echo "  World           : ${WORLD}"
echo "  Batch / K       : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length     : ${CLIP_LEN} frames @ 224x224 (chunk_size=${CHUNK_SIZE}, accum_steps=${ACCUM_STEPS})"
echo "  Optimizer       : ${OPTIMIZER} (weight_decay=${WEIGHT_DECAY})"
echo "  Learning Rates  : Head ${LR}, Backbone $(python -c "print(${LR} * ${BACKBONE_LR_FACTOR})") (${BACKBONE_LR_FACTOR}x)"
echo "  LR Schedule     : CosineAnnealing (warmup=${WARMUP_EPOCHS} ep, delay=${LR_DELAY_EPOCHS} ep, eta_min=${LR_ETA_MIN})"
echo "  Losses          : Cross-Entropy LabelSmooth(0.1) + BatchHard Triplet(margin=${MARGIN})"
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

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/best_model.pth"
    if [ ! -f "${CKPT}" ]; then
        CKPT="${OUTPUT_DIR}/latest_model.pth"
    fi
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Closed-World Test-Set Evaluations for MegaDescriptor (FULL FINE-TUNE)..."
        echo "=========================================================="
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model ${MODEL} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_closed_results/${RUN_NAME}_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model ${MODEL} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_closed_results/${RUN_NAME}_img2vid" \
            --batch_size 32
    fi
    echo "MegaDescriptor (FULL FINE-TUNE Attention) job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
