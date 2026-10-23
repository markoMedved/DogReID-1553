#!/bin/bash
#SBATCH --job-name=eval_diag
#SBATCH --output=log_baseline_results/eval_diagnostics_%j.out
#SBATCH --error=log_baseline_results/eval_diagnostics_%j.err
#SBATCH --time=2:00:00
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

OUTPUT_BASE="evaluation/csvs_diagnostics"
mkdir -p log_baseline_results "${OUTPUT_BASE}"

PYTHON_BIN="/d/hpc/home/mm12755/.conda/envs/project/bin/python"
CKPT="checkpoints_baseline_results/baseline/final_baseline_cos_d25_unfreeze6/latest_model.pth"

echo "=========================================================="
echo "Starting DINOv2 Baseline Diagnostic Evaluations"
echo "  Checkpoint: ${CKPT}"
echo "  Target Dir: ${OUTPUT_BASE}"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint ${CKPT} not found! Exiting..."
    exit 1
fi

# =====================================================================
# PART 1: MASKING & CROP QUALITY DIAGNOSTIC
# =====================================================================
echo ""
echo "##########################################################"
echo "PART 1: MASKING & CROP QUALITY EXPERIMENTS"
echo "##########################################################"

# 1. Video-to-Video YOLO-Masked (Background only)
echo "--> [1/5] V2V: YOLO-Masked (Background-Only Video)"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --mask_dog \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_masked" \
    --batch_size 32

# 2. Image-to-Video Standard (Unmasked, YOLO crop)
echo "--> [2/5] I2V: Standard YOLO Crop (Unmasked)"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --query_images \
    --output_dir "${OUTPUT_BASE}/baseline_img2vid_unmasked_yolo" \
    --batch_size 32

# 3. Image-to-Video GT Crop (Unmasked, Precise Ground-Truth crop on image query)
echo "--> [3/5] I2V: Precise GT Crop (Unmasked Query)"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --query_images \
    --use_gt_for_query_mask \
    --output_dir "${OUTPUT_BASE}/baseline_img2vid_unmasked_gt" \
    --batch_size 32

# 4. Image-to-Video YOLO-Masked (Dog masked using YOLO)
echo "--> [4/5] I2V: YOLO-Masked Query"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --query_images \
    --mask_dog \
    --output_dir "${OUTPUT_BASE}/baseline_img2vid_masked_yolo" \
    --batch_size 32

# 5. Image-to-Video GT-Masked (Dog masked using precise Ground-Truth box)
echo "--> [5/5] I2V: Precise GT-Masked Query"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --query_images \
    --mask_dog \
    --use_gt_for_query_mask \
    --output_dir "${OUTPUT_BASE}/baseline_img2vid_masked_gt" \
    --batch_size 32

# =====================================================================
# PART 2: NUMBER OF FRAMES EXPERIMENT (4, 16, 32 frames)
# =====================================================================
echo ""
echo "##########################################################"
echo "PART 2: NUMBER OF FRAMES EXPERIMENTS (4, 16, 32 frames)"
echo "##########################################################"

# 6. Video-to-Video: 4 Frames
echo "--> [6/8] V2V: 4 Frames"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 4 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len4" \
    --batch_size 32

# 7. Video-to-Video: 16 Frames
echo "--> [7/8] V2V: 16 Frames"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 16 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len16" \
    --batch_size 16

# 8. Video-to-Video: 32 Frames
echo "--> [8/8] V2V: 32 Frames"
${PYTHON_BIN} evaluation/make_csv.py \
    --model baseline \
    --checkpoint "${CKPT}" \
    --world closed \
    --clip_len 32 \
    --output_dir "${OUTPUT_BASE}/baseline_vid2vid_len32" \
    --batch_size 8

# =====================================================================
# PART 3: SUMMARY TABLES GENERATION
# =====================================================================
echo ""
echo "=========================================================="
echo "Compiling Diagnostic Summary Results..."
echo "=========================================================="

${PYTHON_BIN} -c '
import pandas as pd
from pathlib import Path
from evaluation.evaluation_utils import bootstrap_from_csv

OUTPUT_BASE = Path("evaluation/csvs_diagnostics")

def eval_csv(csv_path):
    if not csv_path.exists():
        return "N/A", "N/A", "N/A"
    res = bootstrap_from_csv(str(csv_path), m=100, mode="closed")
    r1 = res["cmc_boot_mean"][0] * 100
    r1_l = res["cmc_ci_lower"][0] * 100
    r1_u = res["cmc_ci_upper"][0] * 100
    m_val = res["mAP_boot_mean"] * 100
    m_l = res["mAP_ci"][0] * 100
    m_u = res["mAP_ci"][1] * 100
    r5 = res["cmc_boot_mean"][4] * 100
    return f"{r1:.2f} [{r1_l:.2f}-{r1_u:.2f}]", f"{m_val:.2f} [{m_l:.2f}-{m_u:.2f}]", f"{r5:.2f}"

# Table 1: Masking
mask_runs = [
    ("V2V Standard (YOLO)", Path("evaluation/csvs_ablations/baseline_vid2vid/closed_dist_matrix.csv")),
    ("V2V YOLO-Masked", OUTPUT_BASE / "baseline_vid2vid_masked" / "closed_dist_matrix.csv"),
    ("I2V Standard (YOLO Crop)", OUTPUT_BASE / "baseline_img2vid_unmasked_yolo" / "closed_dist_matrix.csv"),
    ("I2V GT Crop", OUTPUT_BASE / "baseline_img2vid_unmasked_gt" / "closed_dist_matrix.csv"),
    ("I2V YOLO-Masked", OUTPUT_BASE / "baseline_img2vid_masked_yolo" / "closed_dist_matrix.csv"),
    ("I2V GT-Masked", OUTPUT_BASE / "baseline_img2vid_masked_gt" / "closed_dist_matrix.csv"),
]

print("\n" + "=" * 95)
print("TABLE 1: SUBJECT MASKING & CROP QUALITY DIAGNOSTIC (DINOv2 BASELINE)")
print("=" * 95)
print(f"{\"Condition\":<30} | {\"Rank-1 (%) [95% CI]\":<26} | {\"mAP (%) [95% CI]\":<26} | {\"Rank-5\"}")
print("-" * 95)
for name, p in mask_runs:
    r1_s, map_s, r5_s = eval_csv(p)
    print(f"{name:<30} | {r1_s:<26} | {map_s:<26} | {r5_s}")
print("=" * 95)

# Table 2: Number of Frames
frame_runs = [
    ("4 Frames", OUTPUT_BASE / "baseline_vid2vid_len4" / "closed_dist_matrix.csv"),
    ("8 Frames (Baseline)", Path("evaluation/csvs_ablations/baseline_vid2vid/closed_dist_matrix.csv")),
    ("16 Frames", OUTPUT_BASE / "baseline_vid2vid_len16" / "closed_dist_matrix.csv"),
    ("32 Frames", OUTPUT_BASE / "baseline_vid2vid_len32" / "closed_dist_matrix.csv"),
]

print("\n" + "=" * 95)
print("TABLE 2: TEMPORAL RESOLUTION / NUMBER OF FRAMES (DINOv2 BASELINE)")
print("=" * 95)
print(f"{\"Frame Count\":<30} | {\"Rank-1 (%) [95% CI]\":<26} | {\"mAP (%) [95% CI]\":<26} | {\"Rank-5\"}")
print("-" * 95)
for name, p in frame_runs:
    r1_s, map_s, r5_s = eval_csv(p)
    print(f"{name:<30} | {r1_s:<26} | {map_s:<26} | {r5_s}")
print("=" * 95)
'

echo "Diagnostic evaluations completed successfully!"
