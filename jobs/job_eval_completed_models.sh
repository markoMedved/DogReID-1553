#!/bin/bash
#SBATCH --job-name=eval_comp
#SBATCH --output=logs/eval_completed_%j.out
#SBATCH --error=logs/eval_completed_%j.err
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
mkdir -p logs evaluation/csvs_ablations evaluation/csvs_closed_results evaluation/csvs_open_results

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Evaluation Batch for Completed Models"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# 1. unfreeze2: img2vid
echo "[1/12] Evaluating unfreeze2 (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --representation_type cls \
    --checkpoint checkpoints_ablations/unfreeze2/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_ablations/unfreeze2_img2vid \
    --batch_size 32

# 2. unfreeze8: vid2vid
echo "[2/12] Evaluating unfreeze8 (VID2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --representation_type cls \
    --checkpoint checkpoints_ablations/unfreeze8/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --output_dir evaluation/csvs_ablations/unfreeze8_vid2vid \
    --batch_size 32

# 3. unfreeze8: img2vid
echo "[3/12] Evaluating unfreeze8 (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --representation_type cls \
    --checkpoint checkpoints_ablations/unfreeze8/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_ablations/unfreeze8_img2vid \
    --batch_size 32

# 4. full_finetune: vid2vid
echo "[4/12] Evaluating full_finetune (VID2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --representation_type cls \
    --checkpoint checkpoints_ablations/full_finetune/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --output_dir evaluation/csvs_ablations/full_finetune_vid2vid \
    --batch_size 32

# 5. full_finetune: img2vid
echo "[5/12] Evaluating full_finetune (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --representation_type cls \
    --checkpoint checkpoints_ablations/full_finetune/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_ablations/full_finetune_img2vid \
    --batch_size 32

# 6. closed_clip: vid2vid
echo "[6/12] Evaluating closed_clip (VID2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model clip_baseline \
    --representation_type cls_mean \
    --checkpoint checkpoints/clip_baseline/closed_clip_ep50/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --output_dir evaluation/csvs_closed_results/clip_vid2vid \
    --batch_size 32
mkdir -p evaluation/csvs_ablations/clip_vid2vid
cp evaluation/csvs_closed_results/clip_vid2vid/closed_dist_matrix.csv evaluation/csvs_ablations/clip_vid2vid/closed_dist_matrix.csv 2>/dev/null || true

# 7. closed_clip: img2vid
echo "[7/12] Evaluating closed_clip (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model clip_baseline \
    --representation_type cls_mean \
    --checkpoint checkpoints/clip_baseline/closed_clip_ep50/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_closed_results/clip_img2vid \
    --batch_size 32

# 8. open_clip: vid2vid
echo "[8/12] Evaluating open_clip (VID2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model clip_baseline \
    --representation_type cls_mean \
    --checkpoint checkpoints_open/clip_baseline/open_clip_ep50/latest_model.pth \
    --world open \
    --clip_len 8 \
    --output_dir evaluation/csvs_open_results/clip_vid2vid \
    --batch_size 32

# 9. open_clip: img2vid
echo "[9/12] Evaluating open_clip (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model clip_baseline \
    --representation_type cls_mean \
    --checkpoint checkpoints_open/clip_baseline/open_clip_ep50/latest_model.pth \
    --world open \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_open_results/clip_img2vid \
    --batch_size 32

# 10. closed_md: vid2vid
echo "[10/12] Evaluating closed_md (VID2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model megadescriptor \
    --pooling attention \
    --checkpoint checkpoints/megadescriptor/closed_megadescriptor_ep50/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --output_dir evaluation/csvs_closed_results/megadescriptor_vid2vid \
    --batch_size 32

# 11. closed_md: img2vid
echo "[11/12] Evaluating closed_md (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model megadescriptor \
    --pooling attention \
    --checkpoint checkpoints/megadescriptor/closed_megadescriptor_ep50/latest_model.pth \
    --world closed \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_closed_results/megadescriptor_img2vid \
    --batch_size 32

# 12. open_md: img2vid
echo "[12/12] Evaluating open_md (IMG2VID)..."
${PYTHON_BIN} evaluation/make_csv.py \
    --model megadescriptor \
    --pooling attention \
    --checkpoint checkpoints_open/megadescriptor/open_megadescriptor_ep50/latest_model.pth \
    --world open \
    --clip_len 8 \
    --query_images \
    --output_dir evaluation/csvs_open_results/megadescriptor_img2vid \
    --batch_size 32

echo "=========================================================="
echo "All pending model evaluations finished successfully!"
echo "=========================================================="
