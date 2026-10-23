#!/bin/bash
#SBATCH --job-name=eval_f32
#SBATCH --output=log_baseline_results/eval_frames_32_%j.out
#SBATCH --error=log_baseline_results/eval_frames_32_%j.err
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

OUTPUT_BASE="evaluation/csvs_diagnostics"
PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
CKPT="checkpoints_baseline_results/baseline/final_baseline_cos_d25_unfreeze6/latest_model.pth"

rm -rf ${OUTPUT_BASE}/baseline_vid2vid_len32

# 32 Frames
echo "--> Running 32 Frames Inference..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 32 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len32" \
    --batch_size 8

if [ $? -eq 0 ]; then
    echo "32 Frames evaluation completed successfully!"
else
    echo "Evaluation failed!"
    exit 1
fi
