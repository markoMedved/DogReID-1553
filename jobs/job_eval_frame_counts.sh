#!/bin/bash
#SBATCH --job-name=eval_frames
#SBATCH --output=log_baseline_results/eval_frames_%j.out
#SBATCH --error=log_baseline_results/eval_frames_%j.err
#SBATCH --time=1:00:00
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
CKPT="checkpoints_baseline_results/baseline/final_baseline_cos_d25_unfreeze6/latest_model.pth"

echo "=========================================================="
echo "Starting Variable Frame Count Evaluations (Fixed clip_len)"
echo "  Checkpoint: ${CKPT}"
echo "  Target Dir: ${OUTPUT_BASE}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# Remove old caches for len4, len16, len32 so they recompute fresh
rm -f ${OUTPUT_BASE}/baseline_vid2vid_len4/bootstrap_cache_*.pkl
rm -f ${OUTPUT_BASE}/baseline_vid2vid_len16/bootstrap_cache_*.pkl
rm -f ${OUTPUT_BASE}/baseline_vid2vid_len32/bootstrap_cache_*.pkl

# 1. 4 Frames
echo "--> Running 4 Frames Inference..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 4 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len4" \
    --batch_size 32

# 2. 16 Frames
echo "--> Running 16 Frames Inference..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 16 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len16" \
    --batch_size 16

# 3. 32 Frames
echo "--> Running 32 Frames Inference..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 32 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len32" \
    --batch_size 8

echo "All frame length evaluations completed successfully!"
