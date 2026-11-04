#!/bin/bash
#SBATCH --job-name=ev_clip
#SBATCH --output=logs/eval_clip_%j.out
#SBATCH --error=logs/eval_clip_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -x gwn04

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
mkdir -p logs evaluation/csvs_closed_results evaluation/csvs_open_results evaluation/csvs_ablations

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Dedicated CLIP Re-ID Evaluations"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1. closed_clip: vid2vid
OUT_DIR="evaluation/csvs_closed_results/clip_vid2vid"
if [ -f "${OUT_DIR}/closed_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[1/4] Evaluating closed_clip (VID2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model clip_baseline \
        --representation_type cls_mean \
        --checkpoint checkpoints/clip_baseline/closed_clip_ep50/latest_model.pth \
        --world closed \
        --clip_len 8 \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi
mkdir -p evaluation/csvs_ablations/clip_vid2vid
cp -f evaluation/csvs_closed_results/clip_vid2vid/closed_dist_matrix.csv evaluation/csvs_ablations/clip_vid2vid/closed_dist_matrix.csv 2>/dev/null || true

# 2. closed_clip: img2vid
OUT_DIR="evaluation/csvs_closed_results/clip_img2vid"
if [ -f "${OUT_DIR}/closed_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[2/4] Evaluating closed_clip (IMG2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model clip_baseline \
        --representation_type cls_mean \
        --checkpoint checkpoints/clip_baseline/closed_clip_ep50/latest_model.pth \
        --world closed \
        --clip_len 8 \
        --query_images \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

# 3. open_clip: vid2vid
OUT_DIR="evaluation/csvs_open_results/clip_vid2vid"
if [ -f "${OUT_DIR}/open_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[3/4] Evaluating open_clip (VID2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model clip_baseline \
        --representation_type cls_mean \
        --checkpoint checkpoints_open/clip_baseline/open_clip_ep50/latest_model.pth \
        --world open \
        --clip_len 8 \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

# 4. open_clip: img2vid
OUT_DIR="evaluation/csvs_open_results/clip_img2vid"
if [ -f "${OUT_DIR}/open_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[4/4] Evaluating open_clip (IMG2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model clip_baseline \
        --representation_type cls_mean \
        --checkpoint checkpoints_open/clip_baseline/open_clip_ep50/latest_model.pth \
        --world open \
        --clip_len 8 \
        --query_images \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

echo "=========================================================="
echo "All CLIP evaluations finished successfully!"
echo "=========================================================="
