#!/bin/bash
#SBATCH --job-name=og_sbs
#SBATCH --output=logs_resnet_og_params/resume_sbs_%j.out
#SBATCH --error=logs_resnet_og_params/resume_sbs_%j.err
#SBATCH --time=48:00:00
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

mkdir -p logs_resnet_og_params checkpoints_resnet_og_params evaluation/csvs_ablations

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
OUTPUT_DIR="checkpoints_resnet_og_params/sbs"

echo "=========================================================="
echo "Training OG SBS 60-Epoch Schedule from scratch (FastReID / OpenAnimals Base-SBS)"
echo "  Target Output Dir : ${OUTPUT_DIR}"
echo "  Resolution        : 384x384"
echo "  Target Epochs     : 60 (Cosine delay: 30 epochs)"
echo "  Node              : $(hostname)"
echo "  GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model sbs \
    --reid_method bot \
    --output_dir ${OUTPUT_DIR} \
    --world closed \
    --batch_size 64 \
    --k 4 \
    --clip_len 8 \
    --img_size 384 384 \
    --pooling_type attention \
    --epochs 60 \
    --lr 0.00035 \
    --backbone_lr_factor 1.0 \
    --warmup_epochs 10 \
    --lr_sched cosine \
    --lr_delay_epochs 30 \
    --margin 0.0 \
    --weight_decay 0.0005 \
    --re_prob 0.5 \
    --autoaug_prob 0.1 \
    --val_split 0.0 \
    --eval_period 5 \
    --accum_steps 1 \
    --num_workers 12 \
    --full_finetune \
    --force_yolo \
    --overwrite

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model sbs \
            --checkpoint "${CKPT}" \
            --world closed \
            --output_dir "evaluation/csvs_ablations/og_sbs_vid2vid" \
            --batch_size 32
        ln -sfn og_sbs_vid2vid evaluation/csvs_ablations/sbs_vid2vid
    fi
    echo "SBS 60-epoch training and evaluation complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
