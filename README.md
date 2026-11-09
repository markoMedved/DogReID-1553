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
* **Closed-World Evaluation**: Mean Average Precision (**mAP**), Cumulative Matching Characteristics (**Rank-1**, **Rank-5**) with 95% bootstrap confidence intervals.
* **Open-World Evaluation**: Detection and Identification Rate (**DIR @ FAR = 1%**, **5%**, **10%**) for Rank-1 and Rank-5 with 95% bootstrap confidence intervals across verification distance thresholds.

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
* **📊 Leaderboard:** (Link temporarily omitted during peer-review for double-blind anonymity) — Compare models, inspect confidence intervals, and submit new results.
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

We recommend using **Conda** (Python 3.10 or 3.11) for an isolated, reproducible environment:

```bash
# 1. Create and activate Conda environment
conda create -n dog_reid python=3.11 -y
conda activate dog_reid

# 2. Install PyTorch with CUDA support (adjust cu124 to match your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install all benchmark dependencies
pip install -r requirements.txt
```

> [!NOTE]
> **Pretrained Weights**:
> - **DINOv2**, **CLIP**, **MegaDescriptor**, **RoVF**, and **OpenAnimals ResNet-50** weights are downloaded automatically on first run via `torch.hub`, `timm`, or `huggingface_hub`.
> - For **AdaFreq** and **SCL**, place the ImageNet-pretrained ViT-B/16 checkpoint `jx_vit_base_p16_224-80ecf9dd.pth` at `ECCV-26-SCL/pretrained/jx_vit_base_p16_224-80ecf9dd.pth`.

---

## 3️⃣ Dataset & Repository Directory Structure

Extract the dataset archives into the repository root so the layout matches:

```text
DogReID-1553/
│
├── configs/config.py          # Training configuration & model-specific defaults
├── data/                      # Multi-frame video dataloaders, PK samplers & transforms
├── engine/                    # Specialized trainers (standard Re-ID, AdaFreq, RoVF, SCL)
├── evaluation/                # Evaluation pipeline, distance matrix CSV generator & notebooks
│   ├── evaluate_closed_world.ipynb
│   ├── evaluate_open_world.ipynb
│   ├── evaluate_ablations.ipynb
│   ├── failure_case/
│   │   └── evaluate_identity_and_confusion.ipynb
│   ├── make_csv.py
│   ├── evaluation_utils.py
│   └── run_notebooks.py
├── models/                    # Model factory, wrappers & adapter modules
├── solver/warmup.py           # Warmup schedulers (CosineAnnealingLR, MultiStepLR)
├── tests/                     # Pipeline & preflight verification tests
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
| **`checkpoints_open_validation/`** | **Open-world validation models** (`--world open --val_split 0.2`). | 80% train / 20% validation split |

> [!NOTE]
> **Validation Split vs. Final Benchmark Runs**: By default (`--val_split 0.2`), `train.py` holds out 20% of the training identities for validation monitoring during hyperparameter tuning and ablations. Passing `--val_split 0.0` trains on 100% of the training identities and evaluates directly on the test set (`query` and `gallery`) during the run; this option is **strictly intended for final benchmark runs** once all hyperparameters and design decisions have been fixed.

---

## 🏋️ Training

All 12 benchmark architectures are trained through the unified `train.py` entrypoint. Model-specific defaults (resolution, optimizer, learning rate, schedule, and epochs) are configured automatically in `configs/config.py` when `--model` is specified, and any parameter can be overridden via CLI flags.

All commands below are formatted as **single-line commands** so you can safely copy and paste them directly into Windows PowerShell, Command Prompt, or Linux/macOS bash.

### 1. Foundation Vision Transformers (`baseline` / `dinov2`, `clip_baseline`, `megadescriptor`)

```bash
# DINOv2 ViT-B/14 (Closed-World)
python train.py --model baseline --world closed --pooling_type attention --unfreeze_blocks 6 --epochs 50 --batch_size 32 --lr 2e-5 --lr_sched cosine --lr_delay_epochs 25 --val_split 0.0

# CLIP ViT-B/16 (Closed-World)
python train.py --model clip_baseline --world closed --pooling_type attention --unfreeze_blocks 6 --epochs 50 --batch_size 32 --lr 2e-5 --lr_sched cosine --lr_delay_epochs 25 --val_split 0.0

# MegaDescriptor-B-224 (Closed-World)
python train.py --model megadescriptor --world closed --pooling_type attention --unfreeze_blocks 6 --epochs 50 --batch_size 32 --lr 1e-4 --lr_sched cosine --lr_delay_epochs 25 --val_split 0.0
```

### 2. OpenAnimals ResNet-50 Baselines (`bot`, `agw`, `sbs`, `mgn`, `arbase`)

```bash
# Train any OpenAnimals ResNet-50 baseline (replace --model with: bot | agw | sbs | mgn | arbase)
python train.py --model bot --world closed --pooling_type attention --val_split 0.0
```

### 3. Specialized Animal & Video Re-ID Architectures (`tfclip`, `rovf`, `adafreq`, `scl`)

```bash
# TF-CLIP (AAAI 2024)
python train.py --model tfclip --world closed --val_split 0.0

# RoVF (IJCV 2026: rovf_s or rovf_base)
python train.py --model rovf_s --world closed --epochs 10 --val_split 0.0

# AdaFreq (ECCV 2024)
python train.py --model adafreq --world closed --pooling_type attention --epochs 150 --val_split 0.0

# SCL / MetaN (ECCV 2026)
python train.py --model scl --world closed --pooling_type attention --epochs 60 --val_split 0.0
```

* **Open-World Training**: Add `--world open` to any command above to train on the Open-World split.
* **Ablation Studies**: Customize DINOv2 ablations using `--unfreeze_blocks {0,1,2,4,6,8,10,12}` (or `--full_finetune`), `--pooling_type {attention,mean,max}`, `--clip_len {1,4,8,16,32,64}`, and `--mask_dog`.

---

## 📊 Evaluation

Evaluation is performed in two decoupled stages: **feature extraction & distance matrix calculation** (`evaluation/make_csv.py`), followed by **statistical bootstrap confidence estimation** (interactive Jupyter notebooks or programmatic calls).

### Step 1: Generate Distance Matrix CSV (`evaluation/make_csv.py`)

Use `evaluation/make_csv.py` to extract features, save the query-to-gallery distance matrix CSV, and compute bootstrap metrics.


```bash
# 1. Closed-World Video-to-Video (V2V)
python evaluation/make_csv.py --model baseline --checkpoint checkpoints/baseline/closed_baseline_ep50/latest_model.pth --world closed --output_dir evaluation/csvs_closed_results/baseline_vid2vid

# 2. Closed-World Image-to-Video (I2V: single image query -> video gallery)
python evaluation/make_csv.py --model baseline --checkpoint checkpoints/baseline/closed_baseline_ep50/latest_model.pth --world closed --query_images --output_dir evaluation/csvs_closed_results/baseline_img2vid

# 3. Open-World Video-to-Video (V2V)
python evaluation/make_csv.py --model baseline --checkpoint checkpoints_open/baseline/open_baseline/latest_model.pth --world open --output_dir evaluation/csvs_open_results/baseline_vid2vid

# 4. Open-World Image-to-Video (I2V: single image query -> video gallery)
python evaluation/make_csv.py --model baseline --checkpoint checkpoints_open/baseline/open_baseline/latest_model.pth --world open --query_images --output_dir evaluation/csvs_open_results/baseline_img2vid
```

**Key CLI Options:**
* `--model` / `--model_name`: Architecture identifier (`baseline`/`dinov2`, `clip_baseline`, `megadescriptor`, `bot`, `agw`, `sbs`, `mgn`, `arbase`, `tfclip`, `rovf`, `scl`, `adafreq`).
* `--checkpoint` / `--checkpoint_path`: Direct path to a `.pth` checkpoint or directory. If omitted, `make_csv.py` auto-discovers the checkpoint from `checkpoints/` or `checkpoints_open/`.
* `--world` / `--world_type`: `closed` or `open`.
* `--query_images`: Evaluate single-image queries against the video gallery (Image-to-Video, I2V); omit for Video-to-Video (V2V).
* `--clip_len`: Override the number of sampled frames per video clip during inference (e.g., `1`, `4`, `8`, `16`, `32`, `64`).
* `--mask_dog` / `--use_gt_for_query_mask`: Diagnostic flags for subject masking and ground-truth bounding box crop evaluation.

### Step 2: Calculate Metrics, Tables & Figures (Evaluation Notebooks)

Open the evaluation notebooks (or run `python evaluation/run_notebooks.py --all`) to compute 100-iteration bootstrap confidence intervals and generate all benchmark tables and diagnostic figures:

* **Closed-World Evaluation (`evaluation/evaluate_closed_world.ipynb`)**: Computes and displays the Closed-World V2V and I2V benchmark tables reporting **mAP (%)**, **Rank-1 (%)**, and **Rank-5 (%)** with 95% bootstrap confidence intervals across all 12 models.
* **Open-World Evaluation (`evaluation/evaluate_open_world.ipynb`)**: Computes and displays the Open-World V2V and I2V benchmark tables reporting **Rank-1 DIR** and **Rank-5 DIR** at **FAR = 1%, 5%, and 10%** with 95% bootstrap confidence intervals across all 12 models.
* **Ablation Studies (`evaluation/evaluate_ablations.ipynb`)**: Evaluates and plots (1) **Backbone Adaptation Depth Scaling** (`0, 2, 4, 6, 8, 10, 12` unfrozen transformer blocks), (2) **Subject Masking & Crop Quality Diagnostics** (unmasked vs. YOLO-masked vs. GT-cropped V2V/I2V tables), (3) **Temporal Frame Scaling** (`1, 4, 8, 16, 32, 64` frames table & scaling curve), and (4) **Temporal Aggregation Ablations** (Attention vs. Mean vs. Max pooling table).
* **Failure-Case & Gallery Sensitivity Analysis (`evaluation/failure_case/evaluate_identity_and_confusion.ipynb`)**: Computes per-identity mAP across all 680 closed-world test dogs, generates the **Best-10 vs. Worst-10 Query–Gallery Condition Prevalence** grouped bar chart and video count summary table, evaluates **Mutual & Unidirectional Confusion Pairs** and plots the **Visual Confounding Factors Breakdown** bar chart across the top 20 confused pairs, and plots the **Gallery Size Sensitivity** curves (both Video Volume Scaling and Distractor Identity Scaling).

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
