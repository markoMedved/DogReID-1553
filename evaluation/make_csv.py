import sys
import os
import torch
from pathlib import Path
from collections import OrderedDict

# --- Path Configuration ---
# Make project root visible so imports like "models", "data", etc. work
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

USE_IMAGES = True

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# =================================================================
# --- CONFIGURABLE SETTINGS ---
# =================================================================

# --- Evaluation Environment ---
# closed = all query dogs exist in gallery
# open   = some query dogs are not in gallery
WORLD_TYPE = "closed"

# --- Model Identification ---
# model identifier used for paths and output folders
MODEL_NAME = "swin"

# path to trained checkpoint
MODEL_PATH = str(ROOT_DIR / "trained_models" / f"{MODEL_NAME}_{WORLD_TYPE}" / "model.pth")


# --- MODEL ARCHITECTURE SELECTION ---
# swapping this class switches the entire backbone
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT

MODEL_CLASS = VideoSwin


# --- Output Configuration ---
# where evaluation CSV files will be stored
OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{MODEL_NAME}_{WORLD_TYPE}"
if USE_IMAGES:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{MODEL_NAME}_{WORLD_TYPE}_image"


# name of generated distance matrix
CSV_NAME = f"{WORLD_TYPE}_dist_matrix.csv"

# =================================================================


from data.dataloader import build_test_loaders
from configs.config import Config
from evaluation_utils import (
    generate_distance_csv
)


# --- Setup Configuration Object ---
# create configuration object
cfg = Config()

# specify open/closed world evaluation
cfg.world = WORLD_TYPE

# select device automatically
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# output directory for evaluation files
cfg.output_dir = OUTPUT_FOLDER
cfg.output_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# --- Initialize Model ---
# -------------------------------------------------------------

print(f"-> Initializing Architecture: {MODEL_CLASS.__name__}...")

# create model instance
model = MODEL_CLASS()


print(f"-> Loading Weights: {MODEL_PATH}")

# safety check
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

# --- Load Checkpoint ---
# load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)

# support different checkpoint formats
state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))


# --- Handle DataParallel / DDP Weights ---
# remove "module." prefix if model was trained with DataParallel / DDP
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v


# load weights into model
model.load_state_dict(new_state_dict)

model.to(cfg.device)
model.eval()


# -------------------------------------------------------------
# --- Build Query / Gallery Dataloaders ---
# -------------------------------------------------------------

print(f"-> Preparing {cfg.world.upper()} test dataloaders...")

if USE_IMAGES:
    query_loader, gallery_loader = build_test_loaders(cfg, images=True)

else:
    query_loader, gallery_loader = build_test_loaders(cfg)


# -------------------------------------------------------------
# --- Run Inference and Generate Distance Matrix ---
# -------------------------------------------------------------

print(f"-> Running Inference...")

csv_path = generate_distance_csv(
    model,
    query_loader,
    gallery_loader,
    cfg,
    filename=CSV_NAME
)