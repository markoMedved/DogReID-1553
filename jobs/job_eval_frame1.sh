#!/bin/bash
#SBATCH --job-name=eval_f1
#SBATCH --output=log_baseline_results/eval_frames_1_%j.out
#SBATCH --error=log_baseline_results/eval_frames_1_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

OUTPUT_BASE="evaluation/csvs_diagnostics"
PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
CKPT="checkpoints_baseline_results/baseline/final_baseline_cos_d25_unfreeze6/latest_model.pth"

mkdir -p log_baseline_results "${OUTPUT_BASE}"

echo "=========================================================="
echo "Starting 1-Frame Video-to-Video Evaluation"
echo "  Checkpoint: ${CKPT}"
echo "  Target Dir: ${OUTPUT_BASE}/baseline_vid2vid_len1"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1 Frame
echo "--> Running 1 Frame Inference (batch_size=32)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 1 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len1" \
    --batch_size 32

if [ $? -eq 0 ]; then
    echo "1 Frame evaluation completed successfully!"
else
    echo "Evaluation failed!"
    exit 1
fi
