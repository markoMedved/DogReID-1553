#!/bin/bash
#SBATCH --job-name=ev_md
#SBATCH --output=logs/eval_md_%j.out
#SBATCH --error=logs/eval_md_%j.err
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
mkdir -p logs evaluation/csvs_closed_results evaluation/csvs_open_results

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Dedicated MegaDescriptor Re-ID Evaluations"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1. closed_md: vid2vid
OUT_DIR="evaluation/csvs_closed_results/megadescriptor_vid2vid"
if [ -f "${OUT_DIR}/closed_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[1/3] Evaluating closed_md (VID2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model megadescriptor \
        --pooling attention \
        --checkpoint checkpoints/megadescriptor/closed_megadescriptor_ep50/latest_model.pth \
        --world closed \
        --clip_len 8 \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

# 2. closed_md: img2vid
OUT_DIR="evaluation/csvs_closed_results/megadescriptor_img2vid"
if [ -f "${OUT_DIR}/closed_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[2/3] Evaluating closed_md (IMG2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model megadescriptor \
        --pooling attention \
        --checkpoint checkpoints/megadescriptor/closed_megadescriptor_ep50/latest_model.pth \
        --world closed \
        --clip_len 8 \
        --query_images \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

# 3. open_md: img2vid
OUT_DIR="evaluation/csvs_open_results/megadescriptor_img2vid"
if [ -f "${OUT_DIR}/open_dist_matrix.csv" ]; then
    echo "[SKIP] ${OUT_DIR} already completed."
else
    echo "[3/3] Evaluating open_md (IMG2VID)..."
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model megadescriptor \
        --pooling attention \
        --checkpoint checkpoints_open/megadescriptor/open_megadescriptor_ep50/latest_model.pth \
        --world open \
        --clip_len 8 \
        --query_images \
        --output_dir "${OUT_DIR}" \
        --batch_size 32
fi

echo "=========================================================="
echo "All MegaDescriptor evaluations finished successfully!"
echo "=========================================================="
