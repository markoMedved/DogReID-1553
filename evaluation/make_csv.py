import sys
import os
import torch
import argparse
from pathlib import Path
from collections import OrderedDict
import re

# =================================================================
# --- COMMAND-LINE ARGUMENTS ---
# =================================================================
parser = argparse.ArgumentParser(description="Evaluate Dog Re-ID Models")

parser.add_argument(
    "--model_name", 
    type=str, 
    default="dinov2", 
    choices=["dinov2", "swin", "vit", "bot", "transreid"],
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
    "--query_images",
    action="store_true",
    help="Use images instead of videos for the query set."
)

parser.add_argument(
    "--gallery_images",
    action="store_true",
    help="Use images instead of videos for the gallery set."
)

# --- Checkpoint Identification ---
# These must match the training run, both to locate the checkpoint and to
# rebuild an architecture whose state_dict keys line up.
parser.add_argument(
    "--pooling_type",
    type=str,
    default="attention",
    choices=["attention", "mean", "max"],
    help="Temporal pooling used during training"
)

parser.add_argument(
    "--full_finetune",
    action="store_true",
    help="Set if the checkpoint was trained with full fine-tuning"
)

parser.add_argument(
    "--backbone",
    type=str,
    default=None,
    choices=["dinov2", "osnet", "vit", "swin", "convnext"],
    help="Backbone used during training; defaults to the value in configs/config.py"
)

parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Override the derived checkpoint directory name"
)

args = parser.parse_args()

# --- Assign parsed arguments to variables ---
MODEL_NAME = args.model_name
WORLD_TYPE = args.world_type
QUERY_IMAGES = args.query_images
GALLERY_IMAGES = args.gallery_images

# =================================================================
# --- CONFIGURABLE SETTINGS ---
# =================================================================
# --- Path Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# path to trained checkpoint
# The legacy builders were saved under "{model}_{world}". Runs produced by
# VideoReID use Config.compose_run_name, so the two are derived separately
# rather than assumed equal.
from configs.config import Config as _Config  # noqa: E402

# The backbone is part of the run name, so evaluating an OSNet run requires
# passing --backbone osnet; otherwise the config default is assumed.
BACKBONE = args.backbone or _Config.backbone

if args.run_name is not None:
    RUN_NAME = args.run_name
elif MODEL_NAME in ("bot", "transreid"):
    RUN_NAME = _Config.compose_run_name(
        backbone=BACKBONE,
        reid_method=MODEL_NAME,
        world=WORLD_TYPE,
        pooling_type=args.pooling_type,
        full_finetune=args.full_finetune,
    )
else:
    RUN_NAME = f"{MODEL_NAME}_{WORLD_TYPE}"


# --- Smart Checkpoint Resolution ---
checkpoint_dir = ROOT_DIR / "trained_models" / RUN_NAME
all_checkpoints = list(checkpoint_dir.glob("*.pth"))

max_epoch = -1
latest_ckpt = None

for ckpt in all_checkpoints:
    # Skip 'model.pth' while searching for numbered epoch files
    if ckpt.name == "model.pth":
        continue
        
    # Extract the first sequence of numbers from the filename (e.g., '50' from 'model_50.pth')
    match = re.search(r'(\d+)', ckpt.name)
    if match:
        epoch = int(match.group(1))
        if epoch > max_epoch:
            max_epoch = epoch
            latest_ckpt = ckpt

# Priority 1: Use the checkpoint with the highest epoch number
if latest_ckpt:
    MODEL_PATH = str(latest_ckpt)
    print(f"-> Selected latest epoch checkpoint: {latest_ckpt.name}")
# Priority 2: Fallback to 'model.pth' if no numbered checkpoints exist
elif (checkpoint_dir / "model.pth").exists():
    MODEL_PATH = str(checkpoint_dir / "model.pth")
    print("-> No numbered epoch checkpoints found. Falling back to 'model.pth'.")
# Failsafe: Raise an error if the directory contains no valid .pth files
else:
    raise FileNotFoundError(f"No valid model checkpoints (.pth) found in {checkpoint_dir}")


# --- MODEL ARCHITECTURE SELECTION ---
# swapping this class switches the backbone
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT

if MODEL_NAME == "dinov2":
    MODEL_CLASS = DINOv2ReID
elif MODEL_NAME == "swin":
    MODEL_CLASS = VideoSwin
elif MODEL_NAME == "vit":
    MODEL_CLASS = VideoViT
elif MODEL_NAME in ("bot", "transreid"):
    MODEL_CLASS = None
else:
    raise ValueError("Invalid model name")


# --- Output Configuration ---
# where evaluation CSV files will be stored
modality_str = f"{'img' if QUERY_IMAGES else 'vid'}2{'img' if GALLERY_IMAGES else 'vid'}"
OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{RUN_NAME}_{modality_str}"

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

cfg.model = MODEL_NAME

# FIX: Explicitly set the image size based on the model name here 
# instead of relying on the missing update_model_settings() method.
if "swin" in MODEL_NAME.lower():
    cfg.img_size = (192, 192)
else:
    cfg.img_size = (224, 224)

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

if MODEL_NAME in ("bot", "transreid"):
    from models.reid_model import VideoReID
    print(f"-> Initializing Architecture: VideoReID ({BACKBONE}, {MODEL_NAME}, {args.pooling_type})...")
    cfg.reid_method = MODEL_NAME
    cfg.backbone = BACKBONE
    cfg.model = BACKBONE
    cfg.pooling_type = args.pooling_type

    # The classifiers are unused at inference, but omitting them makes
    # load_state_dict fail on the checkpoint's heads.*.classifier entries.
    # Read the identity count off the checkpoint rather than hardcoding it.
    _ckpt = torch.load(MODEL_PATH, map_location="cpu")
    _sd = _ckpt.get('model', _ckpt.get('state_dict', _ckpt))
    # Match the BNNeck head specifically. Some backbones ship their own
    # classifier, so a bare '*.classifier.weight' match can find the wrong one.
    _cls = [v for k, v in _sd.items()
            if k.startswith("heads.") and k.endswith("classifier.weight")]
    cfg.num_classes = _cls[0].shape[0] if _cls else 0
    print(f"-> num_classes from checkpoint: {cfg.num_classes}")
    del _ckpt, _sd, _cls

    model = VideoReID(cfg)
else:
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

# Pass the updated flags to build_test_loaders
query_loader, gallery_loader = build_test_loaders(
    cfg, 
    query_images=QUERY_IMAGES, 
    gallery_images=GALLERY_IMAGES
)


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