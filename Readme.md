# 🐶 DogReID-1553: Large-Scale Dog Re-Identification Video Dataset

**DogReID-1553** is a large-scale dataset designed for **individual dog
re-identification (Re-ID)** using video data.\
The dataset contains video clips and extracted frames of dogs captured
across different environments, viewpoints, and lighting conditions.

This dataset supports research in:

-   Animal biometrics
-   Lost pet reunification
-   Automated animal welfare monitoring
-   Video-based re-identification systems

The dataset is introduced as part of **Project Puppies**, which aims to
enable new research directions in **animal identity recognition using
computer vision**.

------------------------------------------------------------------------

# 📦 Dataset Overview

DogReID-1553 contains:

-   **1,553 individual dogs**
-   **Video clips (.mp4)** for temporal feature learning
-   **Extracted images (.jpg)** for image-based methods
-   **Train / Query / Gallery splits** provided in `splits.csv`

Each identity appears across **multiple videos and environments**,
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

### Install PyTorch

Install PyTorch compatible with your CUDA version.

Example for **CUDA 12.1**:

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are using **CPU only**:

``` bash
pip install torch torchvision torchaudio
```

------------------------------------------------------------------------

### Install Project Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## 📥 Dataset Download

Download the dataset from:

**\[Dataset Link --- TODO\]**

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
├── job.sh                     # Cluster training job script
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
python train.py
```

------------------------------------------------------------------------

# 📜 Citation

If you use this dataset in your research, please cite:

``` bibtex
TODO
```
