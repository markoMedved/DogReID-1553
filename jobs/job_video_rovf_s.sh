#!/bin/bash
#SBATCH --job-name=v_rovf_s
#SBATCH --output=logs_scl/video_rovf_s_%j.out
#SBATCH --error=logs_scl/video_rovf_s_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -x gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH
export ROVF_CACHE_DIR="/d/hpc/home/mm12755/.cache/rovf_token_cache"

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p logs_scl checkpoints/rovf_s checkpoints/rovf evaluation/csvs_closed_results
rm -rf evaluation/csvs_closed_results/rovf_s_ep5_vid2vid evaluation/csvs_closed_results/rovf_s_ep5_img2vid evaluation/csvs_closed_results/rovf_s_ep10_vid2vid evaluation/csvs_closed_results/rovf_s_ep10_img2vid

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Closed-World RoVF-st (Exact Paper Hyperparameters + CLIP_LEN=8) ---
MODEL="rovf_s"
RUN_NAME="closed_rovf_s_st"
OUTPUT_DIR="checkpoints/rovf_s/${RUN_NAME}"
WORLD="closed"
BATCH_SIZE=30   # 10 mined triplets = 30 clips per step
CLIP_LEN=8      # 8 frames per clip (consistent with other video models)
EPOCHS=10       # Exact RoVF epochs = 10 (5,000 episodes/epoch = 500 steps/ep -> 5,000 total steps)
LR=1e-04
WEIGHT_DECAY=0.01
VAL_SPLIT=0
EVAL_PERIOD=1
NUM_WORKERS=6

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Closed-World Video Training: RoVF-st (Paper Defaults + CLIP_LEN=8)"
echo "  Model        : ${MODEL} (facebook/dinov2-small 384-d + PerceiverV2 384-d, skip_avg=True)"
echo "  Run Name     : ${RUN_NAME}"
echo "  Output Dir   : ${OUTPUT_DIR}"
echo "  World        : ${WORLD}"
echo "  Batch Size   : ${BATCH_SIZE} clips (10 hard-mined triplets/step)"
echo "  Clip Length  : ${CLIP_LEN} frames @ 224x224"
echo "  Epochs       : ${EPOCHS} (5,000 episodes/epoch -> 5,000 total optimizer steps)"
echo "  Scheduler    : WarmupCosineDecay (250 warmup: 1e-3 -> 5e-3, cosine -> 1e-4 by step 4750)"
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
    ln -sfn "../rovf_s/${RUN_NAME}" "checkpoints/rovf/${RUN_NAME}"
    for EP in 5 10; do
        CKPT="${OUTPUT_DIR}/checkpoint_epoch_${EP}.pth"
        if [ -f "${CKPT}" ]; then
            echo "=========================================================="
            echo "Running Closed-World Test-Set Evaluations for RoVF-st (Epoch ${EP})..."
            echo "=========================================================="
            # 1. Closed-World Video-to-Video (Epoch ${EP})
            ${PYTHON_BIN} evaluation/make_csv.py \
                --model rovf_s \
                --checkpoint "${CKPT}" \
                --world closed \
                --clip_len ${CLIP_LEN} \
                --output_dir "evaluation/csvs_closed_results/rovf_s_ep${EP}_vid2vid" \
                --batch_size 32

            # Note: RoVF is strictly a video-to-video method; skipping img2vid evaluation.
        fi
    done
    echo "Closed-World RoVF-st job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
