import sys
import os
import torch
import argparse
from pathlib import Path
from collections import OrderedDict
import re

# =================================================================
# --- PATH CONFIGURATION (Must be before custom imports) ---
# =================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from configs.config import Config as _Config

# =================================================================
# --- COMMAND-LINE ARGUMENTS ---
# =================================================================
parser = argparse.ArgumentParser(description="Evaluate Dog Re-ID Models")

parser.add_argument(
    "--model_name", 
    type=str, 
    default="dinov2", 
    choices=["dinov2", "swin", "vit", "convnetxt", "megadescriptor", "miewid", "bot", "transreid", "tfclip"], 
    help="Model identifier used for paths and architecture selection"
)

parser.add_argument(
    "--world_type", 
    type=str, 
    default="closed", 
    choices=["closed", "open"],
    help="Evaluation environment: 'closed' (all queries in gallery) or 'open' (some not)"
)

# Synchronized with train.py
parser.add_argument(
    "--pooling_type",
    type=str,
    default="attention",
    choices=["attention", "mean", "max", "none", "attn"],
    help="Temporal aggregation method (must match training)"
)

parser.add_argument(
    "--full_finetune",
    action="store_true",
    default=False,
    help="Flag indicating whether backbone was completely fine-tuned during training"
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

parser.add_argument(
    "--use_gt_for_gallery_mask", 
    action="store_true", 
    help="Use Ground Truth bounding boxes specifically for masking the gallery set."
)

# --- BoT / Checkpoint Identification Flags ---
parser.add_argument(
    "--backbone",
    type=str,
    default=None,
    choices=["dinov2", "osnet", "megadescriptor", "vit", "swin", "convnext", "convnetxt"],
    help="Backbone used during training; defaults to the value in configs/config.py"
)

parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Override the derived checkpoint directory name"
)

args = parser.parse_args()

# --- Assign Parsed Arguments ---
BASE_MODEL_NAME = args.model_name
POOLING_TYPE = args.pooling_type
FULL_FINETUNE = args.full_finetune
WORLD_TYPE = args.world_type
MASK_DOG = args.mask_dog
NUM_CLASSES = args.num_classes
BACKBONE = args.backbone or getattr(_Config, "backbone", "dinov2")

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
if args.run_name is not None:
    RUN_NAME = args.run_name
elif BASE_MODEL_NAME in ("bot", "transreid"):
    RUN_NAME = _Config.compose_run_name(
        backbone=BACKBONE,
        reid_method=BASE_MODEL_NAME,
        world=WORLD_TYPE,
        pooling_type=POOLING_TYPE,
        full_finetune=FULL_FINETUNE,
    )
else:
    # Check possible naming schemes in trained_models
    cand_idloss = f"{BASE_MODEL_NAME}_{WORLD_TYPE}_{POOLING_TYPE}_finetune_{FULL_FINETUNE}_idloss_True"
    cand_pool = f"{BASE_MODEL_NAME}_{WORLD_TYPE}_{POOLING_TYPE}_finetune_{FULL_FINETUNE}"
    cand_std = f"{BASE_MODEL_NAME}_{WORLD_TYPE}"
    
    if (ROOT_DIR / "trained_models" / cand_pool).exists():
        RUN_NAME = cand_pool
    elif (ROOT_DIR / "trained_models" / cand_idloss).exists():
        RUN_NAME = cand_idloss
    elif (ROOT_DIR / "trained_models" / cand_std).exists():
        RUN_NAME = cand_std
    else:
        RUN_NAME = cand_pool

# --- Smart Checkpoint Resolution ---
checkpoint_dir = ROOT_DIR / "trained_models" / RUN_NAME
all_checkpoints = list(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []

max_epoch = -1
latest_ckpt = None

for ckpt in all_checkpoints:
    if ckpt.name == "model.pth":
        continue
    match = re.search(r'(\d+)', ckpt.name)
    if match:
        epoch = int(match.group(1))
        if epoch > max_epoch:
            max_epoch = epoch
            latest_ckpt = ckpt

if latest_ckpt:
    MODEL_PATH = str(latest_ckpt)
    print(f"-> Selected latest epoch checkpoint: {latest_ckpt.name}")
elif (checkpoint_dir / "model.pth").exists():
    MODEL_PATH = str(checkpoint_dir / "model.pth")
    print("-> No numbered epoch checkpoints found. Falling back to 'model.pth'.")
elif (ROOT_DIR / "trained_models" / f"{RUN_NAME}.pth").exists():
    MODEL_PATH = str(ROOT_DIR / "trained_models" / f"{RUN_NAME}.pth")
    print(f"-> Found flat checkpoint file at {MODEL_PATH}")
else:
    MODEL_PATH = str(checkpoint_dir / "model.pth")
    print(f"-> Target checkpoint path: {MODEL_PATH}")

# =================================================================
# --- MODEL ARCHITECTURE IMPORT & SELECTION ---
# =================================================================
from models.dinov2_builder import DINOv2ReID
from models.swin_builder import VideoSwin
from models.vit_builder import VideoViT
from models.convnetxt_builder import VideoConvNeXt
from models.megadescriptor_builder import MegaDescriptor
from models.miewid_builder import MiewIDReID

if BASE_MODEL_NAME == "dinov2":
    MODEL_CLASS = DINOv2ReID
elif BASE_MODEL_NAME == "swin":
    MODEL_CLASS = VideoSwin
elif BASE_MODEL_NAME == "vit":
    MODEL_CLASS = VideoViT
elif BASE_MODEL_NAME == "convnetxt":
    MODEL_CLASS = VideoConvNeXt
elif BASE_MODEL_NAME == "megadescriptor":
    MODEL_CLASS = MegaDescriptor
elif BASE_MODEL_NAME == "miewid":
    MODEL_CLASS = MiewIDReID
elif BASE_MODEL_NAME in ("bot", "transreid"):
    MODEL_CLASS = None
else:
    raise ValueError(f"Invalid base model name: {BASE_MODEL_NAME}")

# =================================================================
# --- Output Configuration ---
# =================================================================
suffix = ""
if MASK_DOG:
    suffix += "_masked"

if args.use_gt_for_query_mask and args.use_gt_for_gallery_mask:
    suffix += "_gtboth"
elif args.use_gt_for_query_mask:
    suffix += "_gtq"
elif args.use_gt_for_gallery_mask:
    suffix += "_gtg"

base_folder_name = f"{RUN_NAME}_{MODALITY_TAG}"
if suffix:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / f"{base_folder_name}{suffix}"
    CSV_NAME = f"{suffix.lstrip('_')}_{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"
else:
    OUTPUT_FOLDER = ROOT_DIR / "evaluation" / "csvs" / base_folder_name
    if BASE_MODEL_NAME in ("bot", "transreid"):
        CSV_NAME = f"{WORLD_TYPE}_dist_matrix.csv"
    else:
        CSV_NAME = f"{MODALITY_TAG}_{WORLD_TYPE}_dist_matrix.csv"

from data.dataloader import build_test_loaders
from configs.config import Config
from evaluation_utils import generate_distance_csv

# --- Setup Configuration Object ---
cfg = Config()

cfg.model = BACKBONE if BASE_MODEL_NAME in ("bot", "transreid") else BASE_MODEL_NAME
cfg.backbone = cfg.model

_is_mega = BASE_MODEL_NAME == "megadescriptor" or (
    BASE_MODEL_NAME in ("bot", "transreid") and BACKBONE == "megadescriptor"
)
if _is_mega:
    cfg.img_size = (384, 384) if "384" in getattr(_Config, "megadescriptor_variant", "") else (224, 224)
elif "swin" in cfg.model.lower():
    cfg.img_size = (192, 192)
else:
    cfg.img_size = (224, 224)

cfg.world = WORLD_TYPE
cfg.query_images = QUERY_IMAGES
cfg.gallery_images = GALLERY_IMAGES
cfg.mask_dog = MASK_DOG 
cfg.use_gt_for_query_mask = args.use_gt_for_query_mask
cfg.use_gt_for_gallery_mask = args.use_gt_for_gallery_mask

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.output_dir = OUTPUT_FOLDER
cfg.output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# --- Initialize Model ---
# -------------------------------------------------------------
if BASE_MODEL_NAME in ("bot", "transreid"):
    from models.reid_model import VideoReID
    print(f"-> Initializing Architecture: VideoReID ({BACKBONE}, {BASE_MODEL_NAME}, {POOLING_TYPE})...")
    cfg.reid_method = BASE_MODEL_NAME
    cfg.backbone = BACKBONE
    cfg.model = BACKBONE
    cfg.pooling_type = POOLING_TYPE

    # Read num_classes from checkpoint if available to instantiate classification heads
    if os.path.exists(MODEL_PATH):
        _ckpt = torch.load(MODEL_PATH, map_location="cpu")
        _sd = _ckpt.get('model', _ckpt.get('state_dict', _ckpt))
        _cls = [v for k, v in _sd.items()
                if k.startswith("heads.") and k.endswith("classifier.weight")]
        cfg.num_classes = _cls[0].shape[0] if _cls else 0
        print(f"-> num_classes from checkpoint: {cfg.num_classes}")
        del _ckpt, _sd, _cls

    model = VideoReID(cfg)
elif BASE_MODEL_NAME == "megadescriptor":
    print(f"-> Initializing Architecture: MegaDescriptor ({getattr(_Config, 'megadescriptor_variant', 'default')})...")
    model = MegaDescriptor(variant=getattr(_Config, "megadescriptor_variant", "hf-hub:BVRA/MegaDescriptor-L-224"))
elif BASE_MODEL_NAME == "miewid":
    print("-> Initializing Architecture: MiewIDReID...")
    model = MiewIDReID()
elif BASE_MODEL_NAME == "dinov2":
    print(f"-> Initializing Architecture: DINOv2ReID (Pooling: {POOLING_TYPE}, Num Classes: {NUM_CLASSES})...")
    p_type = "attn" if POOLING_TYPE in ["attention", "attn"] else POOLING_TYPE
    model = DINOv2ReID(
        variant="vitb14_reg",
        num_classes=NUM_CLASSES,
        chunk_size=32,
        pooling_type=p_type
    )
elif BASE_MODEL_NAME in ["vit", "swin"]:
    print(f"-> Initializing Architecture: {MODEL_CLASS.__name__}...")
    try:
        model = MODEL_CLASS(backbone_type="timm")
    except TypeError:
        model = MODEL_CLASS()
elif BASE_MODEL_NAME == "convnetxt":
    print(f"-> Initializing Architecture: VideoConvNeXt...")
    model = VideoConvNeXt()
else:
    model = MODEL_CLASS()

print(f"-> Loading Weights: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if name.startswith("temporal_pool."):
            name = name.replace("temporal_pool.", "temporal_attn.")
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"-> [INFO] Missing keys in state_dict: {len(missing)}")
    if unexpected:
        print(f"-> [INFO] Unexpected keys in state_dict: {len(unexpected)}")
else:
    print(f"-> [WARNING] Checkpoint not found at {MODEL_PATH}. Proceeding with initialized weights.")

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