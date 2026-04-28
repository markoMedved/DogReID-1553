# yolo_pipeline.py
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
ANNOTATIONS_CSV = "bounding_boxes.csv"
IMAGES_ROOT     = "Images"          # folder with DOG_ID subfolders
OUTPUT_DIR      = "yolo_dataset"
BASE_MODEL      = "yolo11n.pt"           # pretrained COCO weights
OUTPUT_NAME     = "yolo11_dogs"
VAL_SPLIT       = 0.15
EPOCHS          = 50
IMGSZ           = 640
BATCH           = 16
DEVICE          = "cuda"
RANDOM_STATE    = 42
# Set to True if your width/height columns are actually x2/y2 coordinates
BBOX_IS_X2Y2    = False

# =============================================================================
# STEP 1 — BBOX CONVERSION
# =============================================================================

def convert_to_yolo(x1, y1, w_or_x2, h_or_y2, img_w, img_h, is_x2y2=False):
    """
    Convert pixel bbox to YOLO normalized (x_center, y_center, w, h).
    Supports both (x1, y1, width, height) and (x1, y1, x2, y2) formats.
    """
    if is_x2y2:
        x_center = ((x1 + w_or_x2) / 2) / img_w
        y_center = ((y1 + h_or_y2) / 2) / img_h
        w_norm   = (w_or_x2 - x1) / img_w
        h_norm   = (h_or_y2 - y1) / img_h
    else:
        x_center = (x1 + w_or_x2 / 2) / img_w
        y_center = (y1 + h_or_y2 / 2) / img_h
        w_norm   = w_or_x2 / img_w
        h_norm   = h_or_y2 / img_h

    # Clamp to [0, 1] to handle annotation overflow
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm   = max(0.0, min(1.0, w_norm))
    h_norm   = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm

# =============================================================================
# STEP 2 — DATASET PREPARATION
# =============================================================================

def prepare_dataset(
    annotations_csv : str,
    images_root     : str,
    output_dir      : str,
    val_split       : float = VAL_SPLIT,
    random_state    : int   = RANDOM_STATE,
    bbox_is_x2y2    : bool  = BBOX_IS_X2Y2,
) -> str:
    """
    Convert CSV annotations to YOLO dataset structure and write dataset.yaml.

    Output structure:
        output_dir/
            images/train/  *.jpg
            images/val/    *.jpg
            labels/train/  *.txt
            labels/val/    *.txt
            dataset.yaml
    Returns:
        Path to dataset.yaml
    """
    print("\n" + "="*60)
    print("STEP 1/2 — Preparing YOLO dataset")
    print("="*60)

    df  = pd.read_csv(annotations_csv)
    out = Path(output_dir)

    # --- Validate required columns ---
    required = {"DOG_ID", "VIDEO_ID", "x_top_left", "y_top_left", "width", "height"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {missing_cols}")

    # --- Directory structure ---
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- Stratified train/val split ---
    # Only stratify on IDs that have more than 1 sample
    counts    = df["DOG_ID"].value_counts()
    valid_ids = counts[counts > 1].index
    df_strat  = df[df["DOG_ID"].isin(valid_ids)]
    df_single = df[~df["DOG_ID"].isin(valid_ids)]  # can't stratify singletons

    train_df, val_df = train_test_split(
        df_strat,
        test_size  = val_split,
        stratify   = df_strat["DOG_ID"],
        random_state = random_state,
    )

    # Singletons go to train only
    train_df = pd.concat([train_df, df_single]).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"  Train samples : {len(train_df)}")
    print(f"  Val   samples : {len(val_df)}")
    if len(df_single) > 0:
        print(f"  Singleton IDs forced to train : {len(df_single)}")

    # --- Process each split ---
    missing_imgs  = 0
    written       = 0

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            dog_id   = row["DOG_ID"]
            video_id = row["VIDEO_ID"]

            src_img = Path(images_root) / dog_id / f"{dog_id}-{video_id}.jpg"

            if not src_img.exists():
                print(f"  [MISSING] {src_img}")
                missing_imgs += 1
                continue

            with Image.open(src_img) as img:
                img_w, img_h = img.size

            x_c, y_c, w_n, h_n = convert_to_yolo(
                row["x_top_left"], row["y_top_left"],
                row["width"],      row["height"],
                img_w, img_h,
                is_x2y2=bbox_is_x2y2,
            )

            # Unique stem avoids collisions between different dogs
            stem = f"{dog_id}__{video_id}"

            shutil.copy2(src_img, out / "images" / split_name / f"{stem}.jpg")

            with open(out / "labels" / split_name / f"{stem}.txt", "w") as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

            written += 1

    print(f"  Written : {written} | Missing : {missing_imgs}")

    # --- dataset.yaml ---
    yaml_content = f"""path: {out.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: dog
"""
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  YAML  : {yaml_path}")

    return str(yaml_path)

# =============================================================================
# STEP 3 — FINE-TUNE
# =============================================================================

def finetune(
    yaml_path   : str,
    base_model  : str = BASE_MODEL,
    output_name : str = OUTPUT_NAME,
    epochs      : int = EPOCHS,
    imgsz       : int = IMGSZ,
    batch       : int = BATCH,
    device      : str = DEVICE,
) -> str:
    """
    Fine-tune YOLOv11 on the prepared dataset.
    Returns path to best weights.
    """
    print("\n" + "="*60)
    print("STEP 2/2 — Fine-tuning YOLOv11")
    print("="*60)

    model = YOLO(base_model)

    model.train(
        data          = yaml_path,
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        device        = device,
        name          = output_name,
        # Fine-tuning hyperparameters
        lr0           = 0.001,
        warmup_epochs = 3,
        mosaic        = 1.0,
        mixup         = 0.1,
        close_mosaic  = 10,    # disable mosaic last 10 epochs for stability
        patience      = 15,    # early stopping
        save_period   = 5,     # checkpoint every 5 epochs
    )

    best = f"runs/detect/{output_name}/weights/best.pt"
    print(f"\n  Best weights : {best}")
    return best

# =============================================================================
# STEP 4 — VALIDATE (optional sanity check)
# =============================================================================

def validate(weights: str, yaml_path: str, imgsz: int = IMGSZ, device: str = DEVICE):
    """Run validation on the val split and print metrics."""
    print("\n" + "="*60)
    print("Validation")
    print("="*60)

    model   = YOLO(weights)
    metrics = model.val(data=yaml_path, imgsz=imgsz, device=device)

    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    return metrics

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # 1 — Prepare dataset
    yaml_path = prepare_dataset(
        annotations_csv = ANNOTATIONS_CSV,
        images_root     = IMAGES_ROOT,
        output_dir      = OUTPUT_DIR,
        bbox_is_x2y2    = BBOX_IS_X2Y2,
    )

    # 2 — Fine-tune
    best_weights = finetune(yaml_path)

    # 3 — Validate
    validate(best_weights, yaml_path)

    print("\n" + "="*60)
    print(f"Done. Use these weights in your ReID pipeline:")
    print(f"  yolo_model='{best_weights}'")
    print("="*60)