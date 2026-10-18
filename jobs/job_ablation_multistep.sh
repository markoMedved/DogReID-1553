#!/bin/bash
#SBATCH --job-name=ab_ms
#SBATCH --output=log_baseline/ablation_multistep_%j.out
#SBATCH --error=log_baseline/ablation_multistep_%j.err
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

mkdir -p log_baseline checkpoints_baseline

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Ablation 3 (Normal LR Drop / MultiStepLR) ---
MODEL="baseline"
REPRESENTATION="cls_gem" # Concat [CLS] + GeM(Patches, p=3.0) -> 1536 dim
RUN_NAME="lr2e-5_ep35_multistep15-25"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=64
K=4
CLIP_LEN=8
EPOCHS=35
LR=2e-05
WEIGHT_DECAY=1e-05
BACKBONE_LR_FACTOR=0.1
UNFREEZE_BLOCKS=2
LR_SCHED="multistep"
VAL_SPLIT=0
EVAL_PERIOD=5
ACCUM_STEPS=1
NUM_WORKERS=12

echo "=========================================================="
echo "Starting Video-to-Video Ablation 3: Normal LR Drop (MultiStepLR)"
echo "  Model           : ${MODEL}"
echo "  Run Name        : ${RUN_NAME}"
echo "  Features        : Concat [CLS] + GeM(Patches, p=3.0) (${REPRESENTATION}, D=1536)"
echo "  LR Scheduler    : Normal Step Decay (--lr_sched multistep, decay 0.1x at ep 15, 25)"
echo "  Learning Rate   : Head ${LR}, Backbone $(python -c "print(${LR} * ${BACKBONE_LR_FACTOR})")"
echo "  Epochs          : ${EPOCHS}"
echo "=========================================================="

python train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --representation_type ${REPRESENTATION} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_sched ${LR_SCHED} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --unfreeze_blocks ${UNFREEZE_BLOCKS} \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --force_yolo

echo "Ablation 3 (Normal LR Drop) complete!"
