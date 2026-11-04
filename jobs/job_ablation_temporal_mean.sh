#!/bin/bash
#SBATCH --job-name=ab_mean
#SBATCH --output=log_ablations/temporal_mean_%j.out
#SBATCH --error=log_ablations/temporal_mean_%j.err
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

mkdir -p log_ablations checkpoints_ablations/temporal_mean evaluation/csvs_ablations

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: DINOv2 Ablation - Temporal Mean Pooling ---
MODEL="baseline"
REID_METHOD="bot"
RUN_NAME="temporal_mean"
OUTPUT_DIR="checkpoints_ablations/${RUN_NAME}"
POOLING="mean"
REPRESENTATION="cls"
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
UNFREEZE_BLOCKS=6
LR_SCHED="cosine"
LR_DELAY_EPOCHS=25
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
LABEL_SMOOTHING=0.1
RE_PROB=0.5
MARGIN=0.3
TRIPLET_TYPE="hard"
VAL_SPLIT=0.0
EVAL_PERIOD=50
ACCUM_STEPS=2
NUM_WORKERS=8

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting DINOv2 Baseline Ablation: Temporal Mean Pooling (50 Epochs)"
echo "  Model           : ${MODEL} (ViT-B/14 Reg, Unfreeze top ${UNFREEZE_BLOCKS} blocks)"
echo "  Pooling         : ${POOLING} (Mean across frames instead of Attention)"
echo "  Representation  : ${REPRESENTATION} (768-d)"
echo "  Output Dir      : ${OUTPUT_DIR}"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --reid_method ${REID_METHOD} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --representation_type ${REPRESENTATION} \
    --pooling_type ${POOLING} \
    --no_full_finetune \
    --unfreeze_blocks ${UNFREEZE_BLOCKS} \
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
        echo "Running Closed-World Evaluations for temporal_mean..."
        echo "=========================================================="
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model baseline \
            --representation_type ${REPRESENTATION} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_ablations/temporal_mean_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model baseline \
            --representation_type ${REPRESENTATION} \
            --pooling ${POOLING} \
            --checkpoint "${CKPT}" \
            --world closed \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_ablations/temporal_mean_img2vid" \
            --batch_size 32
    fi
    echo "Ablation job (temporal_mean) complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
