#!/bin/bash
#SBATCH --job-name=eval_frames
#SBATCH --output=log_baseline_results/eval_frames_%j.out
#SBATCH --error=log_baseline_results/eval_frames_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

OUTPUT_BASE="evaluation/csvs_diagnostics"
PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
CKPT="checkpoints/baseline/closed_baseline_ep50/latest_model.pth"

mkdir -p log_baseline_results "${OUTPUT_BASE}"

echo "=========================================================="
echo "Starting Variable Frame Count Evaluations on DINOv2 Baseline"
echo "  Checkpoint: ${CKPT}"
echo "  Target Dir: ${OUTPUT_BASE}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1. 1 Frame
echo "--> [1/4] Running 1 Frame Inference (batch_size=32)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 1 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len1" \
    --batch_size 32

# 2. 16 Frames
echo "--> [2/4] Running 16 Frames Inference (batch_size=16)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 16 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len16" \
    --batch_size 16

# 3. 32 Frames
echo "--> [3/4] Running 32 Frames Inference (batch_size=8)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 32 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len32" \
    --batch_size 8

# 4. 64 Frames
echo "--> [4/4] Running 64 Frames Inference (batch_size=4)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 64 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len64" \
    --batch_size 4

echo "--> Updating ablations notebook..."
${PYTHON_BIN} evaluation/run_notebooks.py --ablations

echo "All frame length evaluations completed successfully!"
