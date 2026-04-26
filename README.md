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
git clone https://github.com/your-username/DogReID-1553.git
cd DogReID-1553
```

------------------------------------------------------------------------

## ⚙️ Environment Setup

We recommend using **Conda** to create an isolated environment.

### Create Environment

``` bash
conda create -n dog_reid python=3.10 -y
conda activate dog_reid
```

### Install Project Dependencies

    pip install -r requirements.txt


### Install PyTorch

Install PyTorch compatible with your CUDA version.

Example for **CUDA 12.1**:

``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you are using **CPU only**:

``` bash
pip install torch torchvision
```

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

First, open `make_csv.py` and configure the settings at the top of the file to match your trained model:

* **`WORLD_TYPE`**: Set to `"closed"` or `"open"` .
* **`MODEL_NAME`**: A string identifier for your model (e.g., `"dinov2"`, `"vit"`, `"swin"`), for a new model just ignore this, but provide the MODEL_PATH and OUTPUT_FOLDER manually. 
* **`MODEL_PATH`**: The path to your trained model checkpoint (`.pth` file).
* **`MODEL_CLASS`**: Ensure you import and assign the correct architecture class for your weights (e.g., `MODEL_CLASS = DINOv2ReID`).

Once configured, run the script to extract features and generate the distance CSV:

```bash
python make_csv.py
```

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
