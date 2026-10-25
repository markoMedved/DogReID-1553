#!/bin/bash
#SBATCH --job-name=open_sbs
#SBATCH --output=log_open_results/open_sbs_%j.out
#SBATCH --error=log_open_results/open_sbs_%j.err
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
mkdir -p log_open_results checkpoints_open_results evaluation/csvs_open_results

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings (Open-World SBS - ResNet-50-IBN-a + NonLocal + GeMP + CircleSoftmax, 384x384, 35 epochs, CosineAnnealingLR delay 15) ---
MODEL="sbs"
RUN_NAME="open_lr1.2e-4_ep35_cos_d15_bb1x"
OUTPUT_DIR="checkpoints_open_results/sbs/${RUN_NAME}"
POOLING="attention"
WORLD="open"
BATCH_SIZE=64   # P×K = 64 (P=16, K=4)
K=4
CLIP_LEN=8
EPOCHS=35
LR=1.2e-04
DELAY_EPOCHS=15
BACKBONE_LR_FACTOR=1.0
VAL_SPLIT=0
EVAL_PERIOD=5
ACCUM_STEPS=1

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Open-World Video Training: Native SBS (CosineAnnealingLR)"
echo "  Model             : ${MODEL}"
echo "  Run Name          : ${RUN_NAME}"
echo "  Output Dir        : ${OUTPUT_DIR}"
echo "  World             : ${WORLD}"
echo "  Batch / K         : ${BATCH_SIZE} / ${K} (P = $((BATCH_SIZE / K)) identities)"
echo "  Clip Length       : ${CLIP_LEN} frames"
echo "  Learning Rate     : Head ${LR}, Backbone ${LR} (factor: ${BACKBONE_LR_FACTOR})"
echo "  Epochs            : ${EPOCHS} (5-epoch warmup, 5-epoch freeze, cosine delay ${DELAY_EPOCHS})"
echo "  Resolution        : 384x384"
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
    --lr_delay_epochs ${DELAY_EPOCHS} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --full_finetune \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --force_yolo \
    --num_workers 6 \
    --resume

if [ $? -eq 0 ]; then
    CKPT="${OUTPUT_DIR}/latest_model.pth"
    if [ -f "${CKPT}" ]; then
        echo "=========================================================="
        echo "Running Open-World Test-Set Evaluations for SBS..."
        echo "=========================================================="
        # 1. Open-World Video-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model sbs \
            --checkpoint "${CKPT}" \
            --world open \
            --output_dir "evaluation/csvs_open_results/sbs_vid2vid" \
            --batch_size 32

        # 2. Open-World Image-to-Video
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model sbs \
            --checkpoint "${CKPT}" \
            --world open \
            --query_images \
            --output_dir "evaluation/csvs_open_results/sbs_img2vid" \
            --batch_size 32
    fi
    echo "Open-World SBS job complete!"
else
    echo "[ERROR] Training failed! Skipping evaluation."
    exit 1
fi
