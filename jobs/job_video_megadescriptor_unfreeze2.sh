#!/bin/bash
#SBATCH --job-name=v_md_unf2
#SBATCH --output=logs_scl/video_megadescriptor_unfreeze2_%j.out
#SBATCH --error=logs_scl/video_megadescriptor_unfreeze2_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p logs_scl checkpoints/megadescriptor evaluation/csvs_closed_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: MegaDescriptor-B-224 SLIGHTLY FROZEN (unfreeze_blocks=2) ---
MODEL="megadescriptor"
POOLING="attention"
UNFREEZE_BLOCKS=2
RUN_NAME="megadescriptor_b_unfreeze2_${POOLING}_ep51"
OUTPUT_DIR="checkpoints/megadescriptor/${RUN_NAME}"
WORLD="closed"
BATCH_SIZE=32   # P=8, K=4
K=4
CLIP_LEN=8
EPOCHS=51
LR=2e-05        # DINOv2 baseline LR (backbone factor 0.1x = 2e-6)
MARGIN=0.3
WEIGHT_DECAY=1e-05
VAL_SPLIT=0
EVAL_PERIOD=5
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting MegaDescriptor-B-224 Video Training: SLIGHTLY FROZEN"
echo "  Backbone        : ${MODEL} (BVRA/MegaDescriptor-B-224 Swin-B 1024-d)"
echo "  Unfreeze Blocks : ${UNFREEZE_BLOCKS} (last 2 Swin blocks + norm unfrozen)"
echo "  Temporal Pooling: ${POOLING}"
echo "  Run Name        : ${RUN_NAME}"
echo "  Output Dir      : ${OUTPUT_DIR}"
echo "  World           : ${WORLD}"
echo "  Batch / K       : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length     : ${CLIP_LEN} frames @ 224x224"
echo "  Epochs          : ${EPOCHS} (5 warmup epochs, MultiStepLR at 25, 35)"
echo "  Losses          : ID (1.0) + BatchHard Triplet (margin=${MARGIN}, 1.0)"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --pooling_type ${POOLING} \
    --unfreeze_blocks ${UNFREEZE_BLOCKS} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --img_size 224 224 \
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
    CKPT="${OUTPUT_DIR}/best_model.pth"
    if [ ! -f "${CKPT}" ]; then
        CKPT="${OUTPUT_DIR}/latest_model.pth"
    fi
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Closed-World Test-Set Evaluations for MegaDescriptor (UNFREEZE 2)..."
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
    echo "MegaDescriptor (UNFREEZE 2) job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
