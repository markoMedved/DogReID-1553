#!/bin/bash
#SBATCH --job-name=eval_bench
#SBATCH --output=log_baseline/eval_benchmarks_%j.out
#SBATCH --error=log_baseline/eval_benchmarks_%j.err
#SBATCH --time=12:00:00
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

OUTPUT_BASE="evaluation/csvs_closed_results"
mkdir -p log_baseline "${OUTPUT_BASE}" evaluation/csvs_open_results evaluation/csvs_ablations

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"

echo "=========================================================="
echo "Starting Closed-World Benchmark Evaluation"
echo "Target Folder: ${OUTPUT_BASE}"
echo "Running on $(hostname) with GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# Define models and their checkpoints (OpenAnimals ResNet + TF-CLIP)
declare -A MODELS_CKPT
MODELS_CKPT["bot"]="checkpoints_resnet/bot/lr1.2e-4_ep35_ms15-25_bb1x_256x256/latest_model.pth"
MODELS_CKPT["agw"]="checkpoints_resnet/agw/lr1.2e-4_ep35_ms15-25_bb1x/latest_model.pth"
MODELS_CKPT["arbase"]="checkpoints_resnet/arbase/lr3e-5_ep35_cos_d5_bb1x/latest_model.pth"
MODELS_CKPT["sbs"]="checkpoints_resnet/sbs/lr1.2e-4_ep35_cos_d15_bb1x/latest_model.pth"
MODELS_CKPT["mgn"]="checkpoints_resnet/mgn/lr1.2e-4_ep35_cos_d15_bb1x/latest_model.pth"
MODELS_CKPT["tfclip"]="checkpoints_tfclip/tfclip/v1_tfclip_vitb16/latest_model.pth"

MODEL_ORDER=("bot" "agw" "arbase" "sbs" "mgn" "tfclip")

for MODEL in "${MODEL_ORDER[@]}"; do
    CKPT="${MODELS_CKPT[$MODEL]}"
    echo ""
    echo "======================================================================"
    echo "Evaluating Benchmark Model: ${MODEL^^}"
    echo "Checkpoint: ${CKPT}"
    echo "======================================================================"

    if [ ! -f "${CKPT}" ]; then
        echo "[ERROR] Checkpoint ${CKPT} not found! Skipping..."
        continue
    fi

    # 1. Closed-world Video-to-Video
    V2V_DIR="${OUTPUT_BASE}/${MODEL}_vid2vid"
    V2V_CSV="${V2V_DIR}/closed_dist_matrix.csv"
    if [ -f "${V2V_CSV}" ] && [ $(stat -c%s "${V2V_CSV}") -gt 1000000 ]; then
        echo "--> [SKIP] ${V2V_CSV} already exists! Running bootstrap..."
        ${PYTHON_BIN} -c "
from evaluation.evaluation_utils import bootstrap_from_csv
boot_res = bootstrap_from_csv('${V2V_CSV}', m=100, mode='closed')
cmc = boot_res.get('cmc_boot_mean')
map_val = boot_res.get('mAP_boot_mean')
if cmc is not None:
    print(f'Rank-1:  {cmc[0]*100:.2f}% | Rank-5: {cmc[4]*100:.2f}% | Rank-10: {cmc[9]*100:.2f}%')
if map_val is not None:
    print(f'mAP:     {map_val*100:.2f}%')
"
    else
        echo "--> [1/2] ${MODEL^^} Closed-world Video-to-Video"
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model "${MODEL}" \
            --checkpoint "${CKPT}" \
            --world closed \
            --output_dir "${V2V_DIR}" \
            --batch_size 32
    fi

    # 2. Closed-world Image-to-Video
    I2V_DIR="${OUTPUT_BASE}/${MODEL}_img2vid"
    I2V_CSV="${I2V_DIR}/closed_dist_matrix.csv"
    if [ -f "${I2V_CSV}" ] && [ $(stat -c%s "${I2V_CSV}") -gt 1000000 ]; then
        echo "--> [SKIP] ${I2V_CSV} already exists! Running bootstrap..."
        ${PYTHON_BIN} -c "
from evaluation.evaluation_utils import bootstrap_from_csv
boot_res = bootstrap_from_csv('${I2V_CSV}', m=100, mode='closed')
cmc = boot_res.get('cmc_boot_mean')
map_val = boot_res.get('mAP_boot_mean')
if cmc is not None:
    print(f'Rank-1:  {cmc[0]*100:.2f}% | Rank-5: {cmc[4]*100:.2f}% | Rank-10: {cmc[9]*100:.2f}%')
if map_val is not None:
    print(f'mAP:     {map_val*100:.2f}%')
"
    else
        echo "--> [2/2] ${MODEL^^} Closed-world Image-to-Video"
        ${PYTHON_BIN} evaluation/make_csv.py \
            --model "${MODEL}" \
            --checkpoint "${CKPT}" \
            --world closed \
            --query_images \
            --output_dir "${I2V_DIR}" \
            --batch_size 32
    fi

done

echo ""
echo "======================================================================"
echo "Closed-World Benchmark Evaluation Completed!"
echo "All distance matrices saved to ${OUTPUT_BASE}"
echo "======================================================================"

# Print summary table of results
${PYTHON_BIN} -c "
import os, glob
from evaluation.evaluation_utils import bootstrap_from_csv

print('\n' + '='*75)
print(f'CLOSED-WORLD BENCHMARK RESULTS ({OUTPUT_BASE})')
print('='*75)
print(f'{\"Model\":<12} | {\"Modality\":<10} | {\"Rank-1 (%)\":<12} | {\"Rank-5 (%)\":<12} | {\"mAP (%)\":<10}')
print('-'*75)

for m in ['bot', 'agw', 'arbase', 'sbs', 'mgn', 'tfclip']:
    for mod in ['vid2vid', 'img2vid']:
        csv_p = f'${OUTPUT_BASE}/{m}_{mod}/closed_dist_matrix.csv'
        if os.path.exists(csv_p):
            try:
                res = bootstrap_from_csv(csv_p, m=100, mode='closed')
                cmc = res.get('cmc_boot_mean', [0, 0, 0, 0, 0])
                map_v = res.get('mAP_boot_mean', 0)
                print(f'{m.upper():<12} | {mod:<10} | {cmc[0]*100:<12.2f} | {cmc[4]*100:<12.2f} | {map_v*100:<10.2f}')
            except Exception as e:
                print(f'{m.upper():<12} | {mod:<10} | Error: {e}')
print('='*75)
"
