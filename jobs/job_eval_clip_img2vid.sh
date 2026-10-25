#!/bin/bash
#SBATCH --job-name=eval_clip_i2v
#SBATCH --output=log_baseline_results/eval_clip_img2vid_%j.out
#SBATCH --error=log_baseline_results/eval_clip_img2vid_%j.err
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

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
CKPT="checkpoints_baseline_results/baseline/final_ablation_clip_backbone/latest_model.pth"
OUTPUT_DIR="evaluation/csvs_closed_results/clip_baseline_img2vid"

echo "--> Running CLIP Baseline Image-to-Video Inference..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model clip_baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --query_images \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 32

if [ $? -eq 0 ]; then
    ln -sfn clip_baseline_img2vid evaluation/csvs_closed_results/clip_img2vid
    echo "CLIP Image-to-Video evaluation completed successfully!"
else
    echo "Evaluation failed!"
    exit 1
fi
