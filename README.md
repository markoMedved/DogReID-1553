# 🐶 DogReID-1553: Large-Scale Dog Re-Identification Video Dataset and Benchmark

**DogReID-1553** is a large-scale dataset designed for **individual dog re-identification (Re-ID)** using video data.  
The dataset contains video clips and extracted frames of dogs captured across different environments, viewpoints, and lighting conditions.

This dataset supports research in:
* **Animal biometrics**
* **Lost pet reunification**
* **Automated animal welfare monitoring**
* **Video-based re-identification systems**
* **Fine-grained recognition**

The dataset is introduced as part of **Project Puppies**, which aims to enable new research directions in **animal identity recognition using computer vision**. 

------------------------------------------------------------------------

## 📦 Dataset Overview

DogReID-1553 contains:

-   **1,553 individual dogs** and a total of **7463 videos(images)**
    -   **Video clips format: .mp4** 
    -   **Extracted images format: .jpg**
-   **Bounding Boxes**: For the dogs in the first frame of videos / Images dataset provided in `bounding_boxes.csv`.
-   **Train / Query / Gallery splits** provided in `splits.csv`
-   **User provided breeds** in `breeds.csv`

Identities appear across **multiple videos and environments**, making the dataset suitable for **video-based (and image-based) Re-ID benchmarking**.

---

## 🏆 Leaderboard
To track the progress of the community and foster continued innovation in animal biometrics, we maintain an official **DogReID-1553 Leaderboard**. 

We highly encourage researchers, developers, and practitioners to evaluate their novel architectures using our provided evaluation pipeline and submit their results. The leaderboard tracks state-of-the-art performance across both our **Closed-World** (mAP, Rank-1, Rank-5) and **Open-World** (DIR @ FAR) evaluation protocols. 

------------------------------------------------------------------------
## 🚀 Baseline Methods
In this repository, we provide the **benchmark baseline methods** used to evaluate the dataset. This includes the complete training and evaluation pipeline for three state-of-the-art transformer-based architectures (DINOv2, SwinV2, and ViT). By providing the source code of these baselines, we aim to:
1. **Ensure Reproducibility:** Allow researchers to replicate our benchmark results exactly.
2. **Standardize Evaluation:** Provide the official implementation of our Closed-World and Open-World evaluation protocols.
3. **Facilitate Development:** Provide a modular framework that can be easily extended to test new methodologies.

---

## 🔗 Links
* **📄 Paper:** (Under review) — Read the full research paper detailing the creation of DogReID-1553, the methodology, and our baseline findings.
* **📊 Leaderboard:** https://project-puppies.com/leaderboard — View the current state-of-the-art models, compare metrics, and find instructions on how to submit your own model's results.
* **💾 Dataset:** https://doi.org/10.7910/DVN/LVTRLG — Access and download the dataset

------------------------------------------------------------------------
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

------------------------------------------------------------------------


## 📥 Dataset Download

Download the dataset from:

https://doi.org/10.7910/DVN/LVTRLG

After downloading, unzip Videos.zip and Images.zip into:

    DogReID-1553/

Ensure the folders match the structure described below.

------------------------------------------------------------------------
## 📂 Dataset Structure

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

You can generate the distance matrix directly from the terminal using command-line arguments. There is no need to manually edit the script for supported models. Note however that prior to this, you need to have a saved trained model (complete training on the entire training dataset). 

Run `make_csv.py` and configure your run using the following flags:

* **`--model_name`**: The identifier for your architecture (choices: `dinov2`, `swin`, `vit`).
* **`--world_type`**: The evaluation framework to use (choices: `closed` or `open`).
* **`--use_images`**: Include this flag to use images as the query set. Omit it for using videos.

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
---------

### Understanding the Return Values

The `results` dictionary provides different data depending on the `mode` you select. In both modes, the dictionary always includes a **`full_set`** key, which contains the point estimate metrics calculated on the exact original dataset (without bootstrapping). The remaining keys provide the bootstrap statistics. For an example of use, you can check our `closed_set_plots.ipynb` and `open_set_plots.ipynb` notebooks.



#### **Closed-World Setting (`mode="closed"`)**

**1. `full_set` Dictionary (Original Dataset Metrics)**
* **`mAP`**: The overall Mean Average Precision point estimate.
* **`cmc`**: An array containing the exact accuracy at each rank (Rank-1, Rank-2, etc.) for the full dataset.

**2. Bootstrap Statistics (Uncertainty Quantification)**
* **`mAP_boot_mean` / `mAP_std`**: The bootstrap mean and standard deviation for the Mean Average Precision.
* **`mAP_ci`**: A tuple containing the `(lower, upper)` 95% confidence bounds for mAP.
* **`cmc_boot_mean`**: An array containing the mean accuracy at each rank across bootstrap iterations.
* **`cmc_ci_lower` / `cmc_ci_upper`**: Arrays representing the lower and upper 95% confidence boundaries for the CMC curve.


#### **Open-World Setting (`mode="open"`)**

**1. `full_set` Dictionary (Original Dataset Metrics)**
* **`fars`**: An array representing the exact False Alarm Rates (FAR) calculated across all evaluated distance thresholds.
* **`dirs_r1` / `dirs_r5`**: Arrays representing the exact Detection and Identification Rates (DIR) at Rank-1 and Rank-5 across all thresholds.
* **`r1_{target}` / `r5_{target}`** *(e.g., `r1_0.01`, `r5_0.1`)*: The exact DIR scores at specific targeted FAR points (like 1%, 5%, and 10%) for the full dataset. Returns `NaN` if the exact FAR point could not be achieved within the tolerance.

**2. Bootstrap Statistics (Uncertainty Quantification)**
* **`fars_boot_mean`**: The X-axis data array representing the mean False Alarm Rate across bootstrap iterations.
* **`dirs_r1_boot_mean` / `dirs_r5_boot_mean`**: The Y-axis data arrays representing the mean DIR at Rank-1 and Rank-5.
* **`dirs_r1_ci` / `dirs_r5_ci`**: Tuples formatted as `(lower_array, upper_array)` containing the 95% confidence envelopes for the Rank-1 and Rank-5 DIR vs FAR curves.
* **`{target}_boot_stats`** *(e.g., `r1_0.01_boot_stats`, `r5_0.1_boot_stats`)*: Dynamically generated dictionaries for the targeted FAR points. Each dictionary contains:
    * **`mean`**: The bootstrap mean DIR score at that specific FAR.
    * **`ci`**: A tuple `(lower, upper)` for the 95% confidence interval at that specific point.

------------------------------------------------------------------------

# 📜 Citation

If you use this dataset in your research, please cite:

```bibtex
Paper under review
```
