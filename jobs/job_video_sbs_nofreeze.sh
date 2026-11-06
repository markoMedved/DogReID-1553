#!/bin/bash
#SBATCH --job-name=v_sbs_nofz
#SBATCH --output=logs_resnet/video_sbs_nofreeze_%j.out
#SBATCH --error=logs_resnet/video_sbs_nofreeze_%j.err
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
mkdir -p logs_resnet checkpoints_resnet/sbs evaluation/csvs_closed_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: SBS with Default Parameters, NO Freezing at Start ---
MODEL="sbs"
RUN_NAME="lr1.2e-4_ep35_cos_d15_bb1x_nofreeze"
OUTPUT_DIR="checkpoints_resnet/sbs/${RUN_NAME}"
POOLING="attention"
WORLD="closed"
BATCH_SIZE=64   # P×K = 64 (P=16, K=4)
K=4
CLIP_LEN=8
EPOCHS=35
LR=1.2e-04
BACKBONE_LR_FACTOR=1.0
VAL_SPLIT=0          # train on full train split, evaluate on test split
EVAL_PERIOD=5
ACCUM_STEPS=1
NUM_WORKERS=6
FREEZE_EPOCHS=0      # NO FREEZING at the start

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Video Training: SBS (Default Params, NO Freezing)"
echo "  Model             : ${MODEL}"
echo "  Run Name          : ${RUN_NAME}"
echo "  Output Dir        : ${OUTPUT_DIR}"
echo "  Pooling           : ${POOLING}"
echo "  World             : ${WORLD}"
echo "  Batch / K         : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length       : ${CLIP_LEN} frames"
echo "  Learning Rate     : Head ${LR}, Backbone ${LR} (factor: ${BACKBONE_LR_FACTOR})"
echo "  Epochs            : ${EPOCHS} (5-epoch warmup, 0-epoch backbone freeze)"
echo "  Resolution        : 384x384"
echo "  Backbone          : ResNet-50-IBN-a + Non-Local blocks"
echo "  Pooling           : Trainable GeMP (init p=3.0)"
echo "  Classifier        : CircleSoftmax (s=64, m=0.35)"
echo "  Augmentation      : AutoAugment (p=0.1) + REA (p=0.5)"
echo "  Loss              : Soft-margin Triplet (post-BN) + CE"
echo "  Scheduler         : CosineAnnealingLR (15-epoch delay, eta_min 7e-7)"
echo "  Fine-tuning       : Full fine-tuning from epoch 1 (FREEZE_EPOCHS=${FREEZE_EPOCHS})"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lr_sched cosine \
    --lr_delay_epochs 15 \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --full_finetune \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --freeze_epochs ${FREEZE_EPOCHS} \
    --force_yolo \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Closed-World Test-Set Evaluations for SBS (No Freeze)..."
        echo "=========================================================="
        # 1. Closed-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model sbs \
            --checkpoint "${CKPT}" \
            --world closed \
            --output_dir "evaluation/csvs_closed_results/sbs_nofreeze_vid2vid" \
            --batch_size 32

        # 2. Closed-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model sbs \
            --checkpoint "${CKPT}" \
            --world closed \
            --query_images \
            --output_dir "evaluation/csvs_closed_results/sbs_nofreeze_img2vid" \
            --batch_size 32
    fi
    echo "SBS (No Freeze) training and evaluation complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
