import sys
import os
import torch
import argparse
from pathlib import Path
from collections import OrderedDict

# =================================================================
# --- COMMAND-LINE ARGUMENTS ---
# =================================================================
parser = argparse.ArgumentParser(description="Evaluate Dog Re-ID Models")

parser.add_argument(
    "--model_name", 
    type=str, 
    default="dinov2", 
    choices=["dinov2", "swin", "vit"],
    help="Model identifier used for paths and architecture selection"
)

parser.add_argument(
    "--world_type", 
    type=str, 
    default="closed", 
    choices=["closed", "open"],
    help="Evaluation environment: 'closed' (all queries in gallery) or 'open' (some not)"
)

parser.add_argument(
    "--use_images", 
    action="store_true", 
    help="Include this flag to evaluate on images. Omit it to evaluate on videos."
)

args = parser.parse_args()

# --- Assign parsed arguments to variables ---
MODEL_NAME = args.model_name
WORLD_TYPE = args.world_type
USE_IMAGES = args.use_images

# =================================================================
# --- CONFIGURABLE SETTINGS ---
# =================================================================
# --- Path Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

<<<<<<< HEAD
# --- Evaluation Environment ---
# closed = all query dogs exist in gallery
# open   = some query dogs are not in gallery
WORLD_TYPE = "closed"

# --- Model Identification ---
# model identifier used for paths and output folders
MODEL_NAME = "dinov2"
=======
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
>>>>>>> a949c9025cd20c0bc20906e9b28968c8d24f4826

# path to trained checkpoint
MODEL_PATH = str(ROOT_DIR / "trained_models" / f"{MODEL_NAME}_{WORLD_TYPE}" / "model.pth")


# --- MODEL ARCHITECTURE SELECTION ---
# swapping this class switches the backbone
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT

<<<<<<< HEAD
MODEL_CLASS = DINOv2ReID
=======
if MODEL_NAME == "dinov2":
    MODEL_CLASS = DINOv2ReID
elif MODEL_NAME == "swin":
    MODEL_CLASS = VideoSwin
elif MODEL_NAME == "vit":
    MODEL_CLASS = VideoViT
else:
    raise ValueError("Invalid model name")
>>>>>>> a949c9025cd20c0bc20906e9b28968c8d24f4826


# --- Output Configuration ---
# where evaluation CSV files will be stored
if USE_IMAGES:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{MODEL_NAME}_{WORLD_TYPE}_image"
else:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{MODEL_NAME}_{WORLD_TYPE}"


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
checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)

# support different checkpoint formats
state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))


# --- Handle DataParallel / DDP Weights ---
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