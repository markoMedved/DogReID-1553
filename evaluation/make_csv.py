import sys
import os
import torch
import argparse
from pathlib import Path
from collections import OrderedDict

# =================================================================
# --- PATH CONFIGURATION (Must be before custom imports) ---
# =================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

sys.path.append(str(ROOT_DIR))

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

# --- Cross-Modality Flags ---
parser.add_argument(
    "--use_images", 
    action="store_true", 
    help="Shortcut: Sets BOTH query and gallery to images (Image-to-Image)."
)

parser.add_argument(
    "--query_images", 
    action="store_true", 
    help="Evaluate query set using single images instead of video clips."
)

parser.add_argument(
    "--gallery_images", 
    action="store_true", 
    help="Evaluate gallery set using single images instead of video clips."
)

# --- Masking Baseline Flags ---
parser.add_argument(
    "--mask_dog", 
    action="store_true", 
    help="Mask out the dog for the background-only diagnostic baseline."
)

parser.add_argument(
    "--use_gt_for_query_mask", 
    action="store_true", 
    help="Use Ground Truth bounding boxes specifically for masking the query set."
)

args = parser.parse_args()

# --- Assign Parsed Arguments ---
MODEL_NAME = args.model_name
WORLD_TYPE = args.world_type
MASK_DOG = args.mask_dog

# Resolve Query/Gallery Modalities (use_images acts as a master toggle)
QUERY_IMAGES = args.query_images or args.use_images
GALLERY_IMAGES = args.gallery_images or args.use_images

# Descriptive string representation (e.g., "img2vid", "img2img")
q_type = "img" if QUERY_IMAGES else "vid"
g_type = "img" if GALLERY_IMAGES else "vid"
MODALITY_TAG = f"{q_type}2{g_type}"


# =================================================================
# --- CONFIGURABLE SETTINGS ---
# =================================================================
MODEL_PATH = str(ROOT_DIR / "trained_models" / f"{MODEL_NAME}_{WORLD_TYPE}" / "model.pth")


# --- MODEL ARCHITECTURE SELECTION ---
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT

if MODEL_NAME == "dinov2":
    MODEL_CLASS = DINOv2ReID
elif MODEL_NAME == "swin":
    MODEL_CLASS = VideoSwin
elif MODEL_NAME == "vit":
    MODEL_CLASS = VideoViT
else:
    raise ValueError("Invalid model name")


# =================================================================
# --- Output Configuration ---
# =================================================================
base_folder_name = f"{MODEL_NAME}_{WORLD_TYPE}_{MODALITY_TAG}"

if MASK_DOG and args.use_gt_for_query_mask:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{base_folder_name}_masked_gtq"
    CSV_NAME = f"masked_gtq_{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"
elif MASK_DOG:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{base_folder_name}_masked"
    CSV_NAME = f"masked_{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"
elif args.use_gt_for_query_mask:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{base_folder_name}_gtq"
    CSV_NAME = f"gtq_{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"
else:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / base_folder_name
    CSV_NAME = f"{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"


from data.dataloader import build_test_loaders
from configs.config import Config
from evaluation_utils import generate_distance_csv

# --- Setup Configuration Object ---
cfg = Config()

# Sync all flags directly with cfg object
cfg.world = WORLD_TYPE
cfg.query_images = QUERY_IMAGES
cfg.gallery_images = GALLERY_IMAGES
cfg.mask_dog = MASK_DOG 
cfg.use_gt_for_query_mask = args.use_gt_for_query_mask

# Select device automatically
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory for evaluation files
cfg.output_dir = OUTPUT_FOLDER
cfg.output_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# --- Initialize Model ---
# -------------------------------------------------------------
print(f"-> Initializing Architecture: {MODEL_CLASS.__name__}...")

model = MODEL_CLASS()

print(f"-> Loading Weights: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)
state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

# --- Handle DataParallel / DDP Weights ---
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(cfg.device)
model.eval()


# -------------------------------------------------------------
# --- Build Query / Gallery Dataloaders ---
# -------------------------------------------------------------
print(f"-> Preparing {cfg.world.upper()} test dataloaders ({MODALITY_TAG.upper()})...")
print(f"-> Background Masking Baseline: {'ON' if MASK_DOG else 'OFF'}")

# Pass independent modality toggles to the loader generator
query_loader, gallery_loader = build_test_loaders(
    cfg, 
    query_images=QUERY_IMAGES, 
    gallery_images=GALLERY_IMAGES
)


# -------------------------------------------------------------
# --- Run Inference and Generate Distance Matrix ---
# -------------------------------------------------------------
print(f"-> Running Inference ({q_type.upper()} Query -> {g_type.upper()} Gallery)...")

csv_path = generate_distance_csv(
    model,
    query_loader,
    gallery_loader,
    cfg,
    filename=CSV_NAME
)