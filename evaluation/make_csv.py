import sys
import os
import re
import json
import argparse
from pathlib import Path
from collections import OrderedDict
import torch

# =================================================================
# --- PATH CONFIGURATION ---
# =================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from configs.config import Config
from models.model_factory import build_model
from data.dataloader import build_test_loaders
try:
    from evaluation.evaluation_utils import generate_distance_csv, bootstrap_from_csv
except ImportError:
    from evaluation_utils import generate_distance_csv, bootstrap_from_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="DogReID-1553: Easy Distance Matrix Generation & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & Checkpoint
    parser.add_argument(
        "--model", "--model_name",
        dest="model_name",
        type=str,
        default="dinov2",
        help="Model architecture: 'dinov2', 'bot', 'arbase', 'agw', 'sbs', 'mgn', 'tfclip', 'dinov2_bot', etc."
    )
    parser.add_argument(
        "--checkpoint", "--checkpoint_path", "-c",
        dest="checkpoint",
        type=str,
        default=None,
        help="Direct path to checkpoint (.pth) or checkpoint directory. If omitted, auto-discovers."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder name or run tag to search for (e.g., 'lr1.2e-4_ep35_ms15-25_bb1x')."
    )

    # Evaluation Environment
    parser.add_argument(
        "--world", "--world_type",
        dest="world_type",
        type=str,
        default=None,
        choices=["closed", "open"],
        help="Evaluation environment: 'closed' or 'open' (defaults to checkpoint config or 'closed')."
    )
    parser.add_argument(
        "--pooling", "--pooling_type",
        dest="pooling_type",
        type=str,
        default=None,
        choices=["attention", "mean", "max", "none", "attn"],
        help="Temporal aggregation method (defaults to checkpoint config or 'attention')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=None,
        help="Number of frames per video clip (defaults to config value)."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of training identities (auto-detected from checkpoint if 0)."
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        default=None,
        choices=["cls_gem", "cls_mean", "cls", "gem", "patch_mean"],
        help="Feature representation for DINOv2 ('cls_gem', 'cls_mean', 'cls', 'gem'). Auto-detected if not specified."
    )

    # Cross-Modality Flags
    parser.add_argument(
        "--use_images",
        action="store_true",
        help="Shortcut: Sets BOTH query and gallery to single images (Image-to-Image)."
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

    # Bounding Box & Masking Flags
    parser.add_argument(
        "--mask_dog",
        action="store_true",
        help="Mask out the dog for the background-only diagnostic baseline (blacks out inside bounding box)."
    )
    parser.add_argument(
        "--use_gt_boxes", "--use_gt",
        dest="use_gt_boxes",
        action="store_true",
        help="Use Ground Truth bounding boxes from bounding_boxes.csv instead of YOLO for image evaluation."
    )
    parser.add_argument(
        "--use_gt_for_query_mask",
        action="store_true",
        help="Use Ground Truth bounding boxes specifically for the query set."
    )
    parser.add_argument(
        "--use_gt_for_gallery_mask",
        action="store_true",
        help="Use Ground Truth bounding boxes specifically for the gallery set."
    )

    # Output Directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save distance matrix CSV. Defaults to evaluation/csvs/<run_name>_<modality>/."
    )
    parser.add_argument(
        "--skip_bootstrap",
        action="store_true",
        help="Skip automatic bootstrap Rank/mAP calculation after CSV generation."
    )

    return parser.parse_args()


def find_checkpoint_in_dir(directory: Path):
    """Find the best checkpoint file in a directory."""
    if not directory.exists() or not directory.is_dir():
        return None

    # Priority 1: latest_model.pth or best_model.pth
    for priority_name in ("latest_model.pth", "best_model.pth"):
        p = directory / priority_name
        if p.exists():
            return p

    # Priority 2: highest numbered epoch checkpoint
    epoch_ckpts = []
    for ckpt in directory.glob("*.pth"):
        if ckpt.name == "model.pth":
            continue
        match = re.search(r'(\d+)', ckpt.name)
        if match:
            epoch_ckpts.append((int(match.group(1)), ckpt))
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda x: x[0], reverse=True)
        return epoch_ckpts[0][1]

    # Priority 3: model.pth
    if (directory / "model.pth").exists():
        return directory / "model.pth"

    # Priority 4: any .pth file
    all_pth = list(directory.glob("*.pth"))
    if all_pth:
        return all_pth[0]

    return None


def auto_discover_checkpoint(model_name: str, run_name: str = None):
    """
    Intelligently searches across checkpoints_resnet, checkpoints_dinov2, checkpoints_tfclip,
    checkpoints, and trained_models_neurips.
    """
    clean_name = model_name.lower().replace("-", "_")
    search_names = [clean_name]
    if clean_name in ("baseline", "dinov2_baseline"):
        search_names.extend(["dinov2", "dinov2_baseline", "baseline"])
    elif clean_name.startswith("oa_"):
        search_names.append(clean_name[3:])
    elif clean_name.startswith("openanimals_"):
        search_names.append(clean_name[12:])
    else:
        search_names.append(f"oa_{clean_name}")

    candidate_root_dirs = [
        ROOT_DIR / "checkpoints_baseline_results",
        ROOT_DIR / "checkpoints_baseline",
        ROOT_DIR / "checkpoints_resnet",
        ROOT_DIR / "checkpoints_dinov2",
        ROOT_DIR / "checkpoints_dinov2_oa",
        ROOT_DIR / "checkpoints_tfclip",
        ROOT_DIR / "checkpoints_resnet_og_params",
        ROOT_DIR / "checkpoints",
        ROOT_DIR / "trained_models_neurips",
        ROOT_DIR / "trained_models",
    ]

    exact_matches = []
    run_matches = []
    prefix_matches = []

    # Known hybrid suffixes to avoid when searching for base 'dinov2'
    hybrids = {"dinov2_bot", "dinov2_agw", "dinov2_arbase", "dinov2_sbs", "dinov2_mgn"}

    for root in candidate_root_dirs:
        if not root.exists():
            continue

        for s_name in search_names:
            # 1. Exact model directory match (e.g. checkpoints_resnet/bot or trained_models_neurips/oa_bot_...)
            target_sub = root / s_name
            if target_sub.is_dir():
                if run_name:
                    sub_with_run = target_sub / run_name
                    if sub_with_run.is_dir():
                        run_matches.append(sub_with_run)
                else:
                    children = [c for c in target_sub.iterdir() if c.is_dir()]
                    if children:
                        children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        exact_matches.extend(children)
                    exact_matches.append(target_sub)

            # 2. Iterate direct children of root
            for c in root.iterdir():
                if not c.is_dir():
                    continue
                # If searching for pure dinov2, skip hybrid model folders
                if clean_name == "dinov2" and any(c.name.startswith(h) for h in hybrids):
                    continue

                if run_name and run_name in c.name:
                    run_matches.append(c)
                elif c.name == s_name:
                    exact_matches.append(c)
                elif c.name.startswith(f"{s_name}_"):
                    prefix_matches.append(c)

    candidate_dirs = run_matches + exact_matches + prefix_matches

    # Check matched directories for checkpoints (avoiding duplicates)
    seen_dirs = set()
    for d in candidate_dirs:
        resolved = d.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        ckpt = find_checkpoint_in_dir(d)
        if ckpt is not None:
            return ckpt

    return None


def detect_num_classes_from_state_dict(state_dict):
    """Detect identity classification classes if present in checkpoint."""
    classifier_patterns = [
        "heads.classifier.weight",
        "heads.weight",
        "head.classifier.weight",
        "head.weight",
        "classifier.weight",
        "classifier2.weight",
        "classifier_proj.weight",
        "classifier_proj_temp.weight",
        "b1_head.classifier.weight",
        "clip_memory",
    ]
    for k, v in state_dict.items():
        if (any(p in k for p in classifier_patterns) or ("classifier" in k and k.endswith(".weight"))) and hasattr(v, "shape") and len(v.shape) == 2:
            return v.shape[0]
    return 0


def main():
    args = parse_args()

    # --- 1. Checkpoint Resolution ---
    checkpoint_path = None
    checkpoint_dir = None

    if args.checkpoint:
        user_p = Path(args.checkpoint).resolve()
        if user_p.is_file():
            checkpoint_path = user_p
            checkpoint_dir = user_p.parent
        elif user_p.is_dir():
            checkpoint_dir = user_p
            checkpoint_path = find_checkpoint_in_dir(user_p)
        else:
            raise FileNotFoundError(f"Specified checkpoint path does not exist: {user_p}")
    else:
        print(f"-> Auto-discovering checkpoint for model '{args.model_name}'...")
        checkpoint_path = auto_discover_checkpoint(args.model_name, args.run_name)
        if checkpoint_path is not None:
            checkpoint_dir = checkpoint_path.parent

    # --- 2. Load Config from JSON if Available ---
    saved_cfg = {}
    if checkpoint_dir and (checkpoint_dir / "config.json").exists():
        try:
            with open(checkpoint_dir / "config.json", "r") as f:
                saved_cfg = json.load(f)
            print(f"-> Loaded training metadata from {checkpoint_dir / 'config.json'}")
        except Exception as e:
            print(f"-> [INFO] Could not parse config.json: {e}")

    # Determine Model Name
    model_name = args.model_name
    if args.checkpoint and "model" in saved_cfg and args.model_name == "dinov2":
        model_name = saved_cfg["model"]

    # Determine Pooling Type
    pooling_type = args.pooling_type or saved_cfg.get("pooling_type") or "attention"
    if pooling_type == "attn":
        pooling_type = "attention"

    # Determine World Type
    world_type = args.world_type or saved_cfg.get("world") or "closed"

    # Derive Run Name for Folders
    if args.run_name:
        run_name = args.run_name
    elif checkpoint_dir and checkpoint_dir.name != model_name:
        run_name = f"{checkpoint_dir.parent.name}_{checkpoint_dir.name}" if checkpoint_dir.parent.name != ROOT_DIR.name else checkpoint_dir.name
    else:
        run_name = f"{model_name}_{world_type}_{pooling_type}"

    print("==========================================================")
    print("DogReID Distance Matrix Generation & Evaluation")
    print(f"  Model Architecture : {model_name}")
    print(f"  World Mode         : {world_type.upper()}")
    print(f"  Pooling Type       : {pooling_type}")
    print(f"  Checkpoint         : {checkpoint_path if checkpoint_path else 'None (random weights)'}")
    print("==========================================================")

    # --- 3. Configuration Setup ---
    cfg = Config()
    cfg.model = model_name
    cfg.backbone = model_name
    cfg.pooling_type = pooling_type
    cfg.world = world_type
    cfg.test_batch_size = args.batch_size
    target_clip_len = None
    if args.clip_len:
        target_clip_len = args.clip_len
    elif "clip_len" in saved_cfg:
        target_clip_len = saved_cfg["clip_len"]

    # Auto-adjust model settings (sets correct resolution: 384x384 for arbase/mgn, 256x128 for bot/agw, etc.)
    cfg.update_model_settings()

    # Apply clip_len AFTER update_model_settings so it is not overwritten
    if target_clip_len is not None:
        cfg.clip_len = target_clip_len
    print(f"-> Active Clip Length: {cfg.clip_len} frames")

    # If config.json had a specific img_size, respect it
    if "img_size" in saved_cfg:
        cfg.img_size = tuple(saved_cfg["img_size"])

    # Modality Resolution
    query_images = args.query_images or args.use_images
    gallery_images = args.gallery_images or args.use_images
    cfg.query_images = query_images
    cfg.gallery_images = gallery_images
    cfg.mask_dog = args.mask_dog
    cfg.use_gt_for_query_mask = args.use_gt_boxes or args.use_gt_for_query_mask
    cfg.use_gt_for_gallery_mask = args.use_gt_boxes or args.use_gt_for_gallery_mask
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 4. Read State Dict & Auto-Detect num_classes ---
    state_dict = None
    if checkpoint_path and checkpoint_path.exists():
        print(f"-> Reading checkpoint: {checkpoint_path}")
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw_checkpoint.get('model', raw_checkpoint.get('state_dict', raw_checkpoint))

        # Check for num_classes in state_dict
        detected_classes = detect_num_classes_from_state_dict(state_dict)
        if args.num_classes > 0:
            cfg.num_classes = args.num_classes
        elif detected_classes > 0:
            cfg.num_classes = detected_classes
            print(f"-> Auto-detected {cfg.num_classes} identity classes from checkpoint.")
        else:
            cfg.num_classes = 0

        # Auto-detect representation_type, use_bnneck, and fuse_layers for DINOv2 / baseline
        if model_name.lower() in ("dinov2", "baseline", "dinov2_baseline", "clip_baseline", "baseline_clip"):
            bn_dim = None
            has_bn = False
            for k in (
                "bn.weight", "module.bn.weight",
                "heads.0.bn.weight", "module.heads.0.bn.weight",
                "heads.0.bottleneck.weight", "module.heads.0.bottleneck.weight",
                "bottleneck.weight", "module.bottleneck.weight",
            ):
                if k in state_dict and hasattr(state_dict[k], "shape"):
                    bn_dim = state_dict[k].shape[0]
                    has_bn = True
                    break

            if args.representation_type:
                cfg.representation_type = args.representation_type
            elif "representation_type" in saved_cfg:
                cfg.representation_type = saved_cfg["representation_type"]
            else:
                if bn_dim == 768:
                    cfg.representation_type = "cls"
                elif bn_dim == 1536:
                    cfg.representation_type = "cls_gem"

            if "use_bnneck" in saved_cfg:
                cfg.use_bnneck = saved_cfg["use_bnneck"]
            elif not has_bn and any(k in state_dict for k in ("classifier.weight", "heads.0.classifier.weight")):
                cfg.use_bnneck = False

            if "fuse_layers" in saved_cfg:
                cfg.fuse_layers = int(saved_cfg["fuse_layers"])
            elif "backbone.layer_weights" in state_dict:
                cfg.fuse_layers = int(state_dict["backbone.layer_weights"].shape[0])

            cfg.embedding_dim = 1536 if cfg.representation_type in ("cls_mean", "cls_gem") else 768
            print(f"-> DINOv2 representation_type: '{cfg.representation_type}' (embedding_dim={cfg.embedding_dim}, use_bnneck={getattr(cfg, 'use_bnneck', True)}, fuse_layers={getattr(cfg, 'fuse_layers', 1)})")

    # --- 5. Instantiate Architecture via Model Factory ---
    print(f"-> Building model architecture '{model_name}' (resolution={cfg.img_size}, num_classes={cfg.num_classes})...")
    model = build_model(cfg)

    # --- 6. Load Weights ---
    if state_dict is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            if model_name == "dinov2" and name.startswith("temporal_pool."):
                name = name.replace("temporal_pool.", "temporal_attn.")
            new_state_dict[name] = v

        model_state = model.state_dict()
        filtered_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
            if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                if v.shape != model_state[k].shape:
                    print(f"-> [WARNING] Skipping key '{k}' due to shape mismatch: checkpoint {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
                    continue
            filtered_state_dict[k] = v

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"-> [INFO] Missing keys in state_dict: {len(missing)}")
        if unexpected:
            print(f"-> [INFO] Unexpected keys in state_dict: {len(unexpected)}")
        print("-> Weights successfully loaded.")
    else:
        print("-> [WARNING] No checkpoint loaded! Proceeding with initialized weights.")

    model.to(cfg.device)
    model.eval()

    # --- 7. Setup Output Paths ---
    q_type = "img" if query_images else "vid"
    g_type = "img" if gallery_images else "vid"
    modality_tag = f"{q_type}2{g_type}"

    suffix = ""
    if args.mask_dog:
        suffix += "_masked"
    if cfg.use_gt_for_query_mask and cfg.use_gt_for_gallery_mask:
        suffix += "_gtboth"
    elif cfg.use_gt_for_query_mask:
        suffix += "_gtq"
    elif cfg.use_gt_for_gallery_mask:
        suffix += "_gtg"

    base_folder_name = f"{run_name}_{modality_tag}{suffix}"
    if args.output_dir:
        output_folder = Path(args.output_dir)
    else:
        output_folder = ROOT_DIR / "evaluation" / "csvs" / base_folder_name

    output_folder.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = output_folder
    csv_filename = f"{world_type}_dist_matrix.csv"

    # --- 8. Build Query / Gallery Dataloaders ---
    print(f"-> Preparing {world_type.upper()} dataloaders ({modality_tag.upper()}, batch_size={args.batch_size})...")
    query_loader, gallery_loader = build_test_loaders(
        cfg,
        query_images=query_images,
        gallery_images=gallery_images
    )

    # --- 9. Run Inference & Save Distance Matrix ---
    print(f"-> Running inference...")
    csv_path = generate_distance_csv(
        model,
        query_loader,
        gallery_loader,
        cfg,
        filename=csv_filename
    )

    # --- 10. Bootstrap Evaluation ---
    if not args.skip_bootstrap:
        print(f"\n==========================================")
        print(f"Running Bootstrap Evaluation on {csv_path.name} (mode={world_type})...")
        print(f"==========================================")
        boot_res = bootstrap_from_csv(str(csv_path), m=100, mode=world_type)
        if world_type == "closed":
            cmc = boot_res.get("cmc_boot_mean")
            map_val = boot_res.get("mAP_boot_mean")
            if cmc is not None:
                print(f"Rank-1:  {cmc[0]*100:.2f}%")
                print(f"Rank-5:  {cmc[4]*100:.2f}%")
                print(f"Rank-10: {cmc[9]*100:.2f}%")
            if map_val is not None:
                print(f"mAP:     {map_val*100:.2f}%")
        else:
            print("Open-world metrics:")
            for k, v in boot_res.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    main()