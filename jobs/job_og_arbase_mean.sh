#!/bin/bash
#SBATCH --job-name=og_arbase_m
#SBATCH --output=logs_scl/arbase_mean_%j.out
#SBATCH --error=logs_scl/arbase_mean_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=OpenAnimals:$PYTHONPATH

cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553

mkdir -p logs_resnet_og_params checkpoints_resnet_og_params/arbase_mean evaluation/csvs_closed_results evaluation/csvs_ablations

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
OUTPUT_DIR="checkpoints_resnet_og_params/arbase_mean"

echo "=========================================================="
echo "Starting OG OpenAnimals Params Training: ARBase (MEAN Pooling)"
echo "  Target Output Dir : ${OUTPUT_DIR}"
echo "  Pooling           : mean"
echo "  Resolution        : 384x384"
echo "  Target Epochs     : 120 (warmup=15, cosine delay=60, lr=0.00035)"
echo "  Node              : $(hostname)"
echo "  GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model arbase \
    --reid_method bot \
    --output_dir ${OUTPUT_DIR} \
    --world closed \
    --batch_size 64 \
    --k 4 \
    --clip_len 8 \
    --img_size 384 384 \
    --pooling_type mean \
    --epochs 120 \
    --lr 0.00035 \
    --backbone_lr_factor 1.0 \
    --warmup_epochs 15 \
    --lr_sched cosine \
    --lr_delay_epochs 60 \
    --margin 0.3 \
    --weight_decay 0.0005 \
    --re_prob 0.0 \
    --val_split 0.0 \
    --eval_period 5 \
    --accum_steps 1 \
    --num_workers 6 \
    --full_finetune \
    --force_yolo \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model arbase \
            --pooling mean \
            --checkpoint "${CKPT}" \
            --world closed \
            --output_dir "evaluation/csvs_closed_results/og_arbase_mean_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model arbase \
            --pooling mean \
            --checkpoint "${CKPT}" \
            --world closed \
            --query_images \
            --output_dir "evaluation/csvs_closed_results/og_arbase_mean_img2vid" \
            --batch_size 32
    fi
    echo "ARBase (OG OpenAnimals Params, Mean Pooling) training and evaluation complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
