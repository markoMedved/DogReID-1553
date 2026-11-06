#!/bin/bash
#SBATCH --job-name=v_arbase
#SBATCH --output=logs_resnet/video_arbase_%j.out
#SBATCH --error=logs_resnet/video_arbase_%j.err
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

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (Native ARBase - ResNet-50-IBN-a, 384x384, 1.0x LR, CosineAnnealingLR) ---
MODEL="arbase"
RUN_NAME="lr3e-5_ep35_cos_d5_bb1x"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=64   # P×K = 64 (P=16, K=4)
K=4
CLIP_LEN=8
EPOCHS=35
LR=3.0e-05
DELAY_EPOCHS=5
BACKBONE_LR_FACTOR=1.0
VAL_SPLIT=0          # train on full train split, evaluate on test split
EVAL_PERIOD=5
ACCUM_STEPS=1

echo "=========================================================="
echo "Starting Video-to-Video Training: Native ARBase (CosineAnnealingLR)"
echo "  Model             : ${MODEL}"
echo "  Run Name (params) : ${RUN_NAME}"
echo "  Pooling           : ${POOLING}"
echo "  World             : ${WORLD}"
echo "  Batch / K         : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length       : ${CLIP_LEN} frames"
echo "  Learning Rate     : Head ${LR}, Backbone ${LR} (factor: ${BACKBONE_LR_FACTOR})"
echo "  Epochs            : ${EPOCHS} (5-epoch warmup)"
echo "  Resolution        : 384x384"
echo "  REA               : Disabled (re_prob=0.0)"
echo "  Scheduler         : CosineAnnealingLR (${DELAY_EPOCHS}-epoch delay, eta_min 7e-7)"
echo "  Fine-tuning       : Full fine-tuning (backbone LR factor ${BACKBONE_LR_FACTOR}x)"
echo "=========================================================="

python train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lr_sched cosine \
    --lr_delay_epochs ${DELAY_EPOCHS} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --full_finetune \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --force_yolo

echo "Video Native ARBase job complete!"
