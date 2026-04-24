#!/bin/bash
#SBATCH --job-name=dogreid_final_swin_closed
#SBATCH --output=logs_final/swin_closed_%j.out
#SBATCH --error=logs_final/swin_closed_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

echo "Starting Dog Re-ID Swin training job..."

# --- Environment ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate project

# --- Move to project root ---
cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553


# --- Run training ---
python train.py