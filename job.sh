#!/bin/bash
#SBATCH --job-name=swin_lr3e5_m03_wd1e4
#SBATCH --output=logs_swin/train_lr3e5_m03_wd1e4_%j.out
#SBATCH --error=logs_swin/train_lr3e5_m03_wd1e4_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source $(conda info --base)/etc/profile.d/conda.sh
cd /d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553
conda activate project

python train.py --model swin --lr 3e-5 --margin 0.3 --weight_decay 1e-4