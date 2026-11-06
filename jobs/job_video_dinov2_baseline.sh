#!/bin/bash
#SBATCH --job-name=fin_base
#SBATCH --output=log_baseline_results/final_baseline_%j.out
#SBATCH --error=log_baseline_results/final_baseline_%j.err
#SBATCH --time=24:00:00
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

mkdir -p log_baseline_results checkpoints_baseline_results evaluation/csvs_closed_results evaluation/csvs_open_results evaluation/csvs_ablations

echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

# --- Experiment Settings: Final Chosen Baseline ---
# Unfreeze 6 blocks | Pure [CLS] token | BNNeck | Random Erasing | Label Smoothing | Batch-Hard Triplet | Cosine Annealing
MODEL="baseline"
REID_METHOD="bot"
RUN_NAME="final_baseline_cos_d25_unfreeze6"
OUTPUT_DIR="checkpoints_baseline_results/baseline/${RUN_NAME}"
POOLING="attention"
REPRESENTATION="cls"
WORLD="closed"
BATCH_SIZE=32
K=4
CLIP_LEN=8
CHUNK_SIZE=16
EPOCHS=51
LR=2e-05
WEIGHT_DECAY=1e-05
OPTIMIZER="adamw"
BACKBONE_LR_FACTOR=0.1
UNFREEZE_BLOCKS=6
LR_SCHED="cosine"
LR_DELAY_EPOCHS=25
WARMUP_EPOCHS=5
WARMUP_FACTOR=0.01
VAL_SPLIT=0.0
EVAL_PERIOD=51
ACCUM_STEPS=2
NUM_WORKERS=12

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Final Chosen Baseline Training (Full Train Split)"
echo "  Model           : ${MODEL} (VideoReID / reid_method=${REID_METHOD})"
echo "  Run Name        : ${RUN_NAME}"
echo "  Output Dir      : ${OUTPUT_DIR}"
echo "  Backbone        : DINOv2 ViT-B/14 (with registers, pure [CLS])"
echo "  Unfreeze Blocks : ${UNFREEZE_BLOCKS} (top 6 blocks 6..11)"
echo "  Val Split       : ${VAL_SPLIT} (100% train split, test eval at Ep ${EPOCHS})"
echo "  LR Scheduler    : CosineAnnealingLR (delay=${LR_DELAY_EPOCHS})"
echo "  Optimizer       : ${OPTIMIZER} (weight_decay=${WEIGHT_DECAY})"
echo "  Learning Rate   : Head ${LR}, Backbone $(python -c "print(${LR} * ${BACKBONE_LR_FACTOR})")"
echo "=========================================================="

${PYTHON_BIN} train.py \
    --model ${MODEL} \
    --reid_method ${REID_METHOD} \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --representation_type ${REPRESENTATION} \
    --world ${WORLD} \
    --batch_size ${BATCH_SIZE} \
    --k ${K} \
    --clip_len ${CLIP_LEN} \
    --chunk_size ${CHUNK_SIZE} \
    --pooling_type ${POOLING} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --optimizer ${OPTIMIZER} \
    --lr_sched ${LR_SCHED} \
    --lr_delay_epochs ${LR_DELAY_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --warmup_factor ${WARMUP_FACTOR} \
    --val_split ${VAL_SPLIT} \
    --eval_period ${EVAL_PERIOD} \
    --accum_steps ${ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --unfreeze_blocks ${UNFREEZE_BLOCKS} \
    --backbone_lr_factor ${BACKBONE_LR_FACTOR} \
    --force_yolo

CKPT="${OUTPUT_DIR}/latest_model.pth"
if [ -f "${CKPT}" ]; then
    echo "=========================================================="
    echo "Running Final Test-Set Evaluations for Chosen Baseline..."
    echo "=========================================================="
    # 1. Closed-World Video-to-Video
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model baseline \
        --checkpoint "${CKPT}" \
        --world closed \
        --output_dir "evaluation/csvs_closed_results/baseline_vid2vid" \
        --batch_size 32
    mkdir -p "evaluation/csvs_ablations/baseline_vid2vid"
    cp "evaluation/csvs_closed_results/baseline_vid2vid/closed_dist_matrix.csv" "evaluation/csvs_ablations/baseline_vid2vid/closed_dist_matrix.csv"

    # 2. Closed-World Image-to-Video
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model baseline \
        --checkpoint "${CKPT}" \
        --world closed \
        --query_images \
        --output_dir "evaluation/csvs_closed_results/baseline_img2vid" \
        --batch_size 32

    # 3. Open-World Video-to-Video
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model baseline \
        --checkpoint "${CKPT}" \
        --world open \
        --output_dir "evaluation/csvs_open_results/baseline_vid2vid" \
        --batch_size 32

    # 4. Open-World Image-to-Video
    ${PYTHON_BIN} evaluation/make_csv.py \
        --model baseline \
        --checkpoint "${CKPT}" \
        --world open \
        --query_images \
        --output_dir "evaluation/csvs_open_results/baseline_img2vid" \
        --batch_size 32
fi

echo "Final Chosen Baseline job complete!"
