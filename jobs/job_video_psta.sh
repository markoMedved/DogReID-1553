#!/bin/bash
#SBATCH --job-name=v_psta
#SBATCH --output=logs_jobs/video_psta_%j.out
#SBATCH --error=logs_jobs/video_psta_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=gwn04,gwn08

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (PSTA: Pyramid Spatial-Temporal Aggregation) ---
MODEL="psta"
WORLD="closed"
BATCH_SIZE=32   # P×K = 32 (P=8, K=4), matching PSTA paper (8 persons x 4 clips)
K=4
CLIP_LEN=8
EPOCHS=500
LR=3.5e-04
BACKBONE_LR_FACTOR=1.0
VAL_SPLIT=0          # train on full train split, evaluate on test split
EVAL_PERIOD=5        # evaluate every 5 epochs
ACCUM_STEPS=1

# --- Resuming Support ---
RESUME_FLAG=""
if [ "$1" == "--resume" ] || [ "${RESUME}" == "1" ] || [ "${RESUME}" == "true" ] || [ -f "checkpoints/psta/latest_model.pth" ]; then
    echo "Found checkpoint or resume flag requested. Resuming training from checkpoints/psta/latest_model.pth..."
    RESUME_FLAG="--resume"
fi

echo "=========================================================="
echo "Starting Video-to-Video Training: Video PSTA (ICCV 2021)"
echo "  Model             : ${MODEL}"
echo "  World             : ${WORLD}"
echo "  Batch / K         : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length       : ${CLIP_LEN} frames"
echo "  Learning Rate     : Head ${LR}, Backbone ${LR} (factor: ${BACKBONE_LR_FACTOR})"
echo "  Epochs            : ${EPOCHS}"
echo "  Eval Period       : Every ${EVAL_PERIOD} epochs"
echo "  Resolution        : 256x128 (pedestrian crop)"
echo "  Resume            : ${RESUME_FLAG:-None (starting fresh)}"
echo "=========================================================="

python VideoReID-PSTA/Train.py \
    --config_file VideoReID-PSTA/configs/softmax_triplet.yml \
    --dataset dogreid \
    --root_dir . \
    --arch PSTA \
    --seq_len ${CLIP_LEN} \
    SOLVER.MAX_EPOCHS ${EPOCHS} \
    SOLVER.SEQS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.BASE_LR ${LR} \
    SOLVER.EVAL_PERIOD ${EVAL_PERIOD} \
    MODEL.DEVICE_ID "0"

echo "Video PSTA job complete!"
