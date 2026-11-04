#!/bin/bash
#SBATCH --job-name=eval_f1_64
#SBATCH --output=log_baseline_results/eval_frames_1_64_%j.out
#SBATCH --error=log_baseline_results/eval_frames_1_64_%j.err
#SBATCH --time=6:00:00
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
echo "Starting 1-Frame and 64-Frame Video-to-Video Evaluation"
echo "  Checkpoint: ${CKPT}"
echo "  Target Dir: ${OUTPUT_BASE}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1. 1 Frame
echo "--> [1/2] Running 1 Frame Inference (batch_size=32)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 1 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len1" \
    --batch_size 32

# 2. 64 Frames
echo "--> [2/2] Running 64 Frames Inference (batch_size=4)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 64 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len64" \
    --batch_size 4

echo "--> Updating ablations notebook..."
${PYTHON_BIN} evaluation/run_notebooks.py --ablations

echo "1-Frame and 64-Frame evaluations completed successfully!"
