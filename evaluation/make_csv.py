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
    choices=["dinov2", "swin", "vit", "convnetxt", "tfclip"], 
    help="Base model architecture selection"
)

# Synchronized with train.py
parser.add_argument(
    "--pooling_type",
    type=str,
    default="attention",
    choices=["attention", "mean", "max", "none", "attn"],
    help="Temporal aggregation method (must match training)"
)

# Synchronized with train.py
parser.add_argument(
    "--full_finetune",
    action="store_true",
    default=False,
    help="Flag indicating whether backbone was completely fine-tuned during training"
)

parser.add_argument(
    "--world_type", 
    type=str, 
    default="closed", 
    choices=["closed", "open"],
    help="Evaluation environment: 'closed' (all queries in gallery) or 'open' (some not)"
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=0,
    help="Number of training identities (set > 0 if model was trained with an identity classification head)"
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
BASE_MODEL_NAME = args.model_name
POOLING_TYPE = args.pooling_type
FULL_FINETUNE = args.full_finetune  # This is now a boolean to match train.py
WORLD_TYPE = args.world_type
MASK_DOG = args.mask_dog
NUM_CLASSES = args.num_classes

# Resolve Query/Gallery Modalities (use_images acts as a master toggle)
QUERY_IMAGES = args.query_images or args.use_images
GALLERY_IMAGES = args.gallery_images or args.use_images

# Descriptive string representation (e.g., "img2vid", "img2img")
q_type = "img" if QUERY_IMAGES else "vid"
g_type = "img" if GALLERY_IMAGES else "vid"
MODALITY_TAG = f"{q_type}2{g_type}"


# =================================================================
# --- PATH & FOLDER RESOLUTION ---
# =================================================================
MODEL_FOLDER_NAME = f"{BASE_MODEL_NAME}_{WORLD_TYPE}_{POOLING_TYPE}_finetune_{FULL_FINETUNE}"

# Temporarily changed to target model_epoch_10.pth
MODEL_PATH = str(ROOT_DIR / "trained_models" / MODEL_FOLDER_NAME / "model_epoch_10.pth")
base_folder_name = f"{MODEL_FOLDER_NAME}_{MODALITY_TAG}"


# =================================================================
# --- MODEL ARCHITECTURE IMPORT & SELECTION ---
# =================================================================
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT
from models.convnetxt_builder import VideoConvNeXt

if BASE_MODEL_NAME == "dinov2":
    MODEL_CLASS = DINOv2ReID
elif BASE_MODEL_NAME == "swin":
    MODEL_CLASS = VideoSwin
elif BASE_MODEL_NAME == "vit":
    MODEL_CLASS = VideoViT
elif BASE_MODEL_NAME == "convnetxt":
    MODEL_CLASS = VideoConvNeXt
else:
    raise ValueError(f"Invalid base model name: {BASE_MODEL_NAME}")


# =================================================================
# --- Output Configuration ---
# =================================================================
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
cfg.world = WORLD_TYPE
cfg.query_images = QUERY_IMAGES
cfg.gallery_images = GALLERY_IMAGES
cfg.mask_dog = MASK_DOG 
cfg.use_gt_for_query_mask = args.use_gt_for_query_mask

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.output_dir = OUTPUT_FOLDER
cfg.output_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# --- Initialize Model ---
# -------------------------------------------------------------
print(f"-> Initializing Architecture: {MODEL_CLASS.__name__} (Pooling: {POOLING_TYPE}, Num Classes: {NUM_CLASSES})...")

init_kwargs = {}

# Set backbone source for timm-based builders
if BASE_MODEL_NAME in ["vit", "swin"]:
    init_kwargs["backbone_type"] = "timm"

# Set pooling and num_classes for builders that support them (e.g., DINOv2, ViT)
if BASE_MODEL_NAME not in ["swin"]:
    init_kwargs["pooling_type"] = "attn" if POOLING_TYPE == "attention" else POOLING_TYPE
    if NUM_CLASSES > 0:
        init_kwargs["num_classes"] = NUM_CLASSES

try:
    model = MODEL_CLASS(**init_kwargs)
except TypeError:
    # Safe fallback with explicit backbone_type if kwargs fail
    if BASE_MODEL_NAME in ["vit", "swin"]:
        model = MODEL_CLASS(backbone_type="timm")
    else:
        model = MODEL_CLASS()

print(f"-> Loading Weights: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)
state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    
    # Safely remap attention pool naming mismatch between checkpoint and class definition
    if name.startswith("temporal_pool."):
        name = name.replace("temporal_pool.", "temporal_attn.")
        
    new_state_dict[name] = v

model.to(cfg.device)
model.eval()


# -------------------------------------------------------------
# --- Build Query / Gallery Dataloaders ---
# -------------------------------------------------------------
print(f"-> Preparing {cfg.world.upper()} test dataloaders ({MODALITY_TAG.upper()})...")
print(f"-> Background Masking Baseline: {'ON' if MASK_DOG else 'OFF'}")

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