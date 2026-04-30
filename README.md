# 🐶 DogReID-1553: Large-Scale Dog Re-Identification Video Dataset

**DogReID-1553** is a large-scale dataset designed for **individual dog re-identification (Re-ID)** using video data.  
The dataset contains video clips and extracted frames of dogs captured across different environments, viewpoints, and lighting conditions.

This dataset supports research in:
* **Animal biometrics**
* **Lost pet reunification**
* **Automated animal welfare monitoring**
* **Video-based re-identification systems**

The dataset is introduced as part of **Project Puppies**, which aims to enable new research directions in **animal identity recognition using computer vision**. 

---

### 🚀 Baseline Methods
In this repository, we provide the **benchmark baseline methods** used to evaluate the dataset. This includes the complete training and evaluation pipeline for three state-of-the-art transformer-based architectures (DINOv2, SwinV2, and ViT). By providing these baselines, we aim to:
1. **Ensure Reproducibility:** Allow researchers to replicate our benchmark results exactly.
2. **Standardize Evaluation:** Provide the official implementation of our Closed-World and Open-World (DIR@FAR) evaluation protocols.
3. **Facilitate Development:** Provide a modular framework that can be easily extended to test new methodologies.

------------------------------------------------------------------------

# 📦 Dataset Overview

DogReID-1553 contains:

-   **1,553 individual dogs**
-   **Video clips (.mp4)** for temporal feature learning
-   **Extracted images (.jpg)** for image-based methods
-   **Bounding Boxes**: For the dogs in the first frame of videos / Images dataset.
-   **Train / Query / Gallery splits** provided in `splits.csv`

Identities appear across **multiple videos and environments**,
making the dataset suitable for **video-based ReID benchmarking**.

------------------------------------------------------------------------
# 🚀 Quick Start

## 1️⃣ Clone the Repository

``` bash
git clone https://github.com/markoMedved/DogReID-1553.git
cd DogReID-1553
```

------------------------------------------------------------------------

## ⚙️ Environment Setup

We recommend using **Conda** to create an isolated environment.

### Create Environment

```bash
conda create -n dog_reid python=3.10 -y
conda activate dog_reid
```

### Install Base Dependencies

```bash
pip install -r requirements.txt
```

### Install PyTorch (separately)

Install PyTorch based on your system.
Examples:

**CUDA 12.6:**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU only:**

```bash
pip install torch torchvision torchaudio
```

### Install Torch-dependent Libraries

```bash
pip install pytorch-metric-learning timm ultralytics
```

---


------------------------------------------------------------------------


## 📥 Dataset Download

Download the dataset from:

https://doi.org/10.7910/DVN/LVTRLG

After downloading, unzip Videos.zip and Images.zip into:

    DogReID-1553/

Ensure the folders match the structure described above.

------------------------------------------------------------------------
# 📂 Dataset Structure

After downloading and extracting the dataset, the repository should have
the following structure:

```text
DogReID-1553/
│
├── configs/                   # Training configuration files
│   └── config.py
│
├── data/                      # Dataset loading and preprocessing scripts
│
├── engine/                    # Training and optimization logic
│
├── evaluation/                # Evaluation scripts and metrics
│
├── Images/                    # Extracted image frames (.jpg)
│
├── Videos/                    # Video clips (.mp4)
│
├── models/                    # Model architectures
│
├── bounding_boxes.csv         # Bounding box annotations
│
├── breeds.csv                 # Dog breed metadata
│
├── splits.csv                 # Train / Query / Gallery splits
│
├── train.py                   # Main training script
│
├── requirements.txt           # Python dependencies
│
└── README.md
```
------------------------------------------------------------------------

## 🏋️ Training

Training parameters can be modified inside:

    configs/config.py

### Start Training

``` bash
python train.py  # Can also specify parameters here
```
------------------------------------------------------------------------


## 📊 Evaluation

Evaluating a trained model is a two-step process: generating a distance matrix CSV, and then running bootstrap sampling to calculate the final metrics.

### 1. Generate the Distance Matrix

You can generate the distance matrix directly from the terminal using command-line arguments. There is no need to manually edit the script for supported models. 

Run `make_csv.py` and configure your run using the following flags:

* **`--model_name`**: The identifier for your architecture (choices: `dinov2`, `swin`, `vit`).
* **`--world_type`**: The evaluation framework to use (choices: `closed` or `open`).
* **`--use_images`**: Include this flag to evaluate on static images. Omit it to evaluate on video clips.

**Example Command:**
```bash
python make_csv.py --model_name dinov2 --world_type open --use_images
```
This will run inference to extract the features and automatically save a distance matrix CSV to `evaluation/csvs/<model_name>_<world_type>/`.

> **Note for Custom Architectures:** If you are evaluating a brand-new model architecture not included in the default parser choices, you will need to open `make_csv.py` to manually define your `MODEL_CLASS` and provide the exact `MODEL_PATH` and `OUTPUT_FOLDER`.

This will save a distance matrix CSV to `evaluation/csvs/<MODEL_NAME>_<WORLD_TYPE>/`.

### 2. Calculate Metrics (Bootstrap Evaluation)

Once the distance matrix is ready, use the bootstrap evaluation to calculate statistically robust metrics. This process resamples the data with replacement to provide mean scores and **95% Confidence Intervals**.

```python
from evaluation_utils import bootstrap_from_csv

# Path to the CSV generated in Step 1
csv_file = "evaluation/csvs/dinov2_closed/closed_dist_matrix.csv"

# m=100 is recommended for stable confidence intervals
results = bootstrap_from_csv(csv_path=csv_file, m=100, mode="closed")
```


### Understanding the Return Values

The `results` dictionary provides different data depending on the `mode` you select.

#### **Closed-World Setting (`mode="closed"`)**
Used when every query dog is known to exist in the gallery.
* **`mAP_mean` / `mAP_std`**: The average precision and its standard deviation.
* **`cmc_mean`**: An array containing the mean accuracy at each rank (Rank-1, Rank-2, etc.).
* **`cmc_lower` / `cmc_upper`**: The 95% confidence boundaries for the CMC curve.
* **`ranks`**: An array of integers $[1, 2, 3, ...]$ for easy plotting.

#### **Open-World Setting (`mode="open"`)**
Used when the query set contains "stranger" dogs not present in the gallery.
* **`mean_fars`**: The X-axis data (False Accept Rate).
* **`mean_dirs`**: The Y-axis data (Detection and Identification Rate).
* **`lower_dirs` / `upper_dirs`**: The 95% confidence "envelope" for the DIR curve.
* **`targets`**: A dictionary containing the specific DIR scores at exactly **1%**, **5%**, and **10%** FAR.

------------------------------------------------------------------------

# 📜 Citation

If you use this dataset in your research, please cite:

```bibtex
TODO
```
