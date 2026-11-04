# 🐶 DogReID-1553: Large-Scale Dog Re-Identification Video Dataset and Benchmark

**DogReID-1553** is a large-scale video dataset designed for **individual dog re-identification (Re-ID)** in real-world scenarios.  
The dataset contains video sequences and extracted image frames of dogs captured across diverse outdoor and indoor environments, varying viewpoints, natural poses, and dynamic lighting conditions.

This dataset supports research in:
* **Animal biometrics and fine-grained recognition**
* **Lost pet identification and reunification**
* **Automated wildlife and animal welfare monitoring**
* **Multi-frame video-based and static image-to-video re-identification**

The dataset is introduced as part of **Project Puppies**, which aims to advance animal biometrics using modern computer vision and deep metric learning.

---

## 📦 Dataset Overview

DogReID-1553 contains:

- **1,553 individual dogs** across **7,463 video clips** (and corresponding extracted frame images)
  - **Video clips format**: `.mp4` (8-frame temporal video sequences for video-to-video retrieval)
  - **Static images format**: `.jpg` (bounding-box crops for image-to-video retrieval)
- **Bounding Boxes**: Ground-truth bounding boxes for dogs across frames provided in `bounding_boxes.csv`.
- **Identity Splits**: Closed-World and Open-World train, query, and gallery splits provided in `splits.csv`.
- **Breed Annotations**: User-verified dog breed metadata provided in `breeds.csv`.

Identities appear across **multiple cameras, dates, and backgrounds**, enabling rigorous benchmarking in both **Video-to-Video (V2V)** and **Image-to-Video (I2V)** settings.

---

## 🏆 Leaderboard

To track community progress and foster reproducible innovation in animal biometrics, we host the official **DogReID-1553 Leaderboard**.

Researchers are encouraged to evaluate their architectures using our standardized evaluation protocols:
* **Closed-World Evaluation**: Mean Average Precision (**mAP**), Cumulative Matching Characteristics (**Rank-1**, **Rank-5**, **Rank-10**) with 95% bootstrap confidence intervals.
* **Open-World Evaluation**: Detection and Identification Rate (**DIR @ FAR = 1%**, **5%**, **10%**) across varying verification distance thresholds.

---

## 🚀 Benchmark Baselines & Architectures

This repository provides reproducible implementations for all baseline architectures evaluated in our benchmark:

1. **Baselines using vision foundational models**:
   - **DINOv2** (ViT-B/14 with registers): Self-supervised transformer with top-block adaptation and temporal attention pooling.
   - **CLIP** (ViT-B/16): Contrastively pre-trained vision-language foundation backbone.
   - **MegaDescriptor** (MegaDescriptor-B-224): Wildlife-foundation model pre-trained on diverse animal species.
2. **Specialized Animal & Video Re-ID Frameworks**:
   - **TF-CLIP** (*AAAI 2024*): Text-free CLIP video Re-ID with sequence-specific prompts and temporal memory diffusion.
   - **RoVF** (*IJCV 2026*): Recurrence over Video Frames using Perceiver-based temporal recurrence on top of DINOv2.
   - **AdaFreq** (*ECCV 2024*): Adaptive high-frequency transformer isolating species textures from wilderness background clutter.
   - **SCL / MetaN** (*ECCV 2026*): Cross-species Animal Re-Identification with Semantic Consistency Learning.
3. **ResNet-50 Baselines from OpenAnimals** (*ICCV 2025*):
   - **BoT** (Bag of Tricks for Re-ID)
   - **AGW** (Attention Generalized Weighting with non-local attention and generalized mean pooling)
   - **SBS** (Stronger Baseline with Circle Loss and cosine decay)
   - **MGN** (Multiple Granularity Network with multi-branch stripe pooling)
   - **ARBase** (Animal Re-ID Base tailored for animal biometrics)

---

## 🔗 Links

* **📄 Paper:** *(Under review)* — Full research paper describing dataset curation, identity verification protocols, and extensive benchmark evaluations.
* **📊 Leaderboard:** https://project-puppies.com/leaderboard — Compare models, inspect confidence intervals, and submit new results.
* **💾 Dataset Download:** *(Link temporarily omitted during peer-review for double-blind anonymity)*.

---

# 🚀 Quick Start

## 1️⃣ Clone the Repository

```bash
git clone [repository_name]
cd DogReID-1553
```

---

## 2️⃣ Environment Setup

We recommend using **Conda** for an isolated, reproducible environment.

```bash
# Create and activate environment
conda create -n dogreid python=3.10 -y
conda activate dogreid

# Install PyTorch with CUDA support (e.g. CUDA 12.4/12.6)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install Core Dependencies
pip install -r requirements.txt

# Install Metric Learning and Detection Packages
pip install pytorch-metric-learning==2.9.0 ultralytics==8.4.41 timm
```

---

## 3️⃣ Dataset Directory Structure

Extract the dataset archives into the repository root so the layout matches:

```text
DogReID-1553/
│
├── configs/                   # Training configuration definitions (configs/config.py)
├── data/                      # Multi-frame video dataloaders & transforms
├── engine/                    # Specialized trainers (DINOv2, AdaFreq, RoVF, SCL)
├── evaluation/                # Evaluation pipeline, distance matrix CSV generator & notebooks
│   ├── evaluate_closed_world.ipynb
│   ├── evaluate_open_world.ipynb
│   ├── evaluate_ablations.ipynb
│   ├── make_csv.py
│   └── evaluation_utils.py
├── models/                    # Model factory, wrappers & adapter modules
├── losses/                    # Batch-hard triplet, soft triplet, and cross-entropy losses
├── solver/                    # Warmup schedulers (CosineAnnealingLR, MultiStepLR) & optimizers
├── jobs/                      # HPC SLURM batch submission scripts
│
├── bounding_boxes.csv         # Dog bounding box annotations
├── breeds.csv                 # Dog breed annotations
├── splits.csv                 # Closed-world and open-world identity splits
│
├── Images/                    # Extracted static frame images (.jpg)
├── Videos/                    # Video clip sequences (.mp4)
│
├── train.py                   # Unified training script for all architectures
├── requirements.txt           # Python package requirements
└── README.md
```

---

## 📁 Checkpoints Organization

Trained model checkpoints are automatically organized into distinct directories based on `--world` (`closed` vs. `open`) and `--val_split`:

| Directory | Purpose | Training Set |
| :--- | :--- | :--- |
| **`checkpoints/`** | **Final closed-world benchmark models** (`--val_split 0.0`). | 100% of closed-world training identities |
| **`checkpoints_validation/`** | **Closed-world hyperparameter & ablation models** (`--val_split 0.2`, default). | 80% train / 20% validation split |
| **`checkpoints_open/`** | **Final open-world benchmark models** (`--world open --val_split 0.0`). | 100% of open-world training identities |
| **`checkpoints_experiments/`** | **Historical archives** of exploratory runs and parameter sweeps. | Archival |

> [!NOTE]
> **Validation Split vs. Final Benchmark Runs**: By default (`--val_split 0.2`), `train.py` holds out 20% of the training identities for validation monitoring during hyperparameter tuning and ablations. Passing `--val_split 0.0` trains on 100% of the training identities and evaluates directly on the test set (`query` and `gallery`) during the run; this option is **strictly intended for final benchmark runs** once all hyperparameters and design decisions have been fixed.

---

## 🏋️ Training

All architectures are trained through the unified `train.py` entrypoint. Model-specific defaults (resolution, optimizer, learning rate, schedule, and epochs) are configured automatically in `configs/config.py` when `--model` is specified, and any parameter can be overridden via CLI flags.

### 1. Foundation Vision Transformers (`baseline` / `dinov2`, `clip_baseline`, `megadescriptor`)
```bash
# DINOv2 ViT-B/14 (or replace --model with clip_baseline or megadescriptor)
python train.py \
    --model baseline \
    --pooling_type attention \
    --unfreeze_blocks 6 \
    --epochs 51 \
    --batch_size 32 \
    --lr 2e-5 \
    --lr_sched cosine \
    --lr_delay_epochs 25 \
    --val_split 0.0
```

### 2. OpenAnimals ResNet-50 Baselines (`bot`, `agw`, `sbs`, `mgn`, `arbase`)
```bash
# Train any OpenAnimals ResNet-50 baseline (bot, agw, sbs, mgn, arbase)
python train.py \
    --model bot \
    --pooling_type attention \
    --val_split 0.0
```

### 3. Specialized Animal & Video Re-ID Architectures (`tfclip`, `rovf`, `adafreq`, `scl`)
```bash
# TF-CLIP (AAAI 2024)
python train.py --model tfclip --pooling_type attention --val_split 0.0

# RoVF (IJCV 2026: rovf_s or rovf_base)
python train.py --model rovf_s --epochs 10 --val_split 0.0

# AdaFreq (ECCV 2024)
python train.py --model adafreq --pooling_type attention --epochs 150 --val_split 0.0

# SCL / MetaN (ECCV 2026)
python train.py --model scl --pooling_type attention --epochs 60 --val_split 0.0
```

---

## 📊 Evaluation

Evaluation is performed in two decoupled stages: **feature extraction & distance matrix calculation** (`evaluation/make_csv.py`), followed by **statistical bootstrap confidence estimation** (interactive Jupyter notebooks or programmatic calls).

### Step 1: Generate Distance Matrix CSV

Use `evaluation/make_csv.py` to run inference across queries and galleries:

```bash
# Closed-World Video-to-Video Evaluation (DINOv2)
python evaluation/make_csv.py \
    --model_name baseline \
    --world_type closed

# Closed-World Image-to-Video Evaluation (Static Query Crop)
python evaluation/make_csv.py \
    --model_name baseline \
    --world_type closed \
    --use_images

# Open-World Video-to-Video Evaluation
python evaluation/make_csv.py \
    --model_name baseline \
    --world_type open
```

**Key CLI Options:**
* `--model_name`: Architecture identifier (`baseline`, `clip`, `megadescriptor`, `bot`, `agw`, `sbs`, `mgn`, `arbase`, `tfclip`, `rovf`, `scl`, `adafreq`).
* `--checkpoint_path`: Explicit path to a `latest_model.pth`. If omitted, the script automatically resolves the checkpoint from `checkpoints/`.
* `--world_type`: `closed` or `open`.
* `--use_images`: Query modality — set for Image-to-Video (I2V); omit for Video-to-Video (V2V).
* `--chunk_size`: Max frames processed simultaneously per GPU forward pass (defaults to 16 for memory efficiency).

Distance matrices and metadata are saved under `evaluation/csvs_<world_type>_results/<model>_<modality>/`.

### Step 2: Calculate Metrics & 95% Confidence Intervals

Open the dedicated evaluation notebooks to calculate bootstrap metrics and generate publication-ready tables and plots:

* **Closed-World Evaluation**: `evaluation/evaluate_closed_world.ipynb`
* **Open-World Evaluation**: `evaluation/evaluate_open_world.ipynb`
* **Ablation Studies**: `evaluation/evaluate_ablations.ipynb`

Alternatively, run bootstrap calculation programmatically:

```python
from evaluation.evaluation_utils import bootstrap_from_csv

# Closed-world bootstrap (m=100 resamples)
results = bootstrap_from_csv(
    csv_path="evaluation/csvs_closed_results/baseline_vid2vid/closed_dist_matrix.csv",
    m=100,
    mode="closed"
)

print(f"mAP: {results['mAP_boot_mean']*100:.1f} ± {results['mAP_std']*100:.1f}%")
print(f"Rank-1: {results['cmc_boot_mean'][0]*100:.1f}% [95% CI: {results['cmc_ci_lower'][0]*100:.1f}-{results['cmc_ci_upper'][0]*100:.1f}%]")
```

---

## 🙏 Acknowledgments & References

We gratefully acknowledge the authors and open-source contributors of the following works and codebases integrated into our benchmark:

- **[RoVF](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification)** — Rogers et al., *"Recurrence over Video Frames (RoVF) for Animal Re-identification"*, *International Journal of Computer Vision* (IJCV), 2026.
- **[ECCV-26 SCL](https://github.com/Kemalau/ECCV-26-SCL)** — Chen, Li, & Ye, *"Cross-species Animal Re-identification with Semantic Consistency Learning"*, *European Conference on Computer Vision* (ECCV), 2026.
- **[AdaFreq](https://github.com/JigglypuffStitch/AdaFreq)** — Li, Chen, & Ye, *"Adaptive high-frequency transformer for diverse wildlife re-identification"*, *European Conference on Computer Vision* (ECCV), 2024.
- **[TF-CLIP](https://github.com/AsuradaYuci/TF-CLIP)** — Yu et al., *"Tf-clip: Learning text-free clip for video-based person re-identification"*, *Proceedings of the AAAI Conference on Artificial Intelligence* (AAAI), 2024.
- **[OpenAnimals](https://github.com/hshustc/OpenAnimals)** — Hou et al., *"Openanimals: Revisiting person re-identification for animals towards better generalization"*, *IEEE/CVF International Conference on Computer Vision* (ICCV), 2025.

---

## 📜 Citation

If you find DogReID-1553 or this codebase helpful in your research, please cite:

```bibtex
Paper currently under review
```
