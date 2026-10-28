#!/bin/bash
#SBATCH --job-name=open_rovf
#SBATCH --output=log_open_results/open_rovf_%j.out
#SBATCH --error=log_open_results/open_rovf_%j.err
#SBATCH --time=24:00:00
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
mkdir -p log_open_results checkpoints_open_results/rovf evaluation/csvs_open_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Open-World RoVF-S-ST (Exact Original RoVF Hyperparameters) ---
MODEL="rovf"
RUN_NAME="open_rovf_s_st"
OUTPUT_DIR="checkpoints_open_results/rovf/${RUN_NAME}"
WORLD="open"
BATCH_SIZE=30   # 10 mined triplets = 30 clips per step
CLIP_LEN=5      # Exact RoVF num_frames = 5
EPOCHS=10       # Exact RoVF epochs = 10 (5,000 episodes/epoch = 500 steps/ep -> 5,000 total steps)
LR=1e-04
WEIGHT_DECAY=0.01
VAL_SPLIT=0
EVAL_PERIOD=1
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Open-World Video Training: RoVF-S-ST"
echo "  Model        : ${MODEL} (facebook/dinov2-small + PerceiverV2 + avg skip)"
echo "  Run Name     : ${RUN_NAME}"
echo "  Output Dir   : ${OUTPUT_DIR}"
echo "  World        : ${WORLD}"
echo "  Batch Size   : ${BATCH_SIZE} clips (10 hard-mined triplets/step)"
echo "  Clip Length  : ${CLIP_LEN} frames @ 224x224"
echo "  Epochs       : ${EPOCHS} (5,000 episodes/epoch -> 5,000 total optimizer steps)"
echo "  Scheduler    : WarmupCosineDecay (250 warmup: 1e-3 -> 5e-3, 4750 decay -> 1e-4)"
echo "  Criterion    : TripletMarginLoss(margin=1.0, p=2)"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --clip_len ${CLIP_LEN} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --num_workers ${NUM_WORKERS} \
    --force_yolo

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Open-World Test-Set Evaluations for RoVF-S-ST..."
        echo "=========================================================="
        # 1. Open-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model rovf \
            --checkpoint "${CKPT}" \
            --world open \
            --clip_len ${CLIP_LEN} \
            --output_dir "evaluation/csvs_open_results/rovf_vid2vid" \
            --batch_size 32

        # 2. Open-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model rovf \
            --checkpoint "${CKPT}" \
            --world open \
            --clip_len ${CLIP_LEN} \
            --query_images \
            --output_dir "evaluation/csvs_open_results/rovf_img2vid" \
            --batch_size 32
    fi
    echo "Open-World RoVF-S-ST job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
