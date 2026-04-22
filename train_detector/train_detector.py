import os
import pandas as pd
import shutil
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURATION ---
BBOX_CSV = 'C:/Users/marko/Desktop/DogReID-1553/bounding_boxes.csv'
SPLIT_CSV = 'C:/Users/marko/Desktop/DogReID-1553/splits.csv'
IMAGES_ROOT = 'C:/Users/marko/Desktop/DogReID-1553/Images' 
OUTPUT_DIR = 'yolo_dataset'

# --- SPLIT MODE SELECTION ---
# Set this to 'SPLIT_CLOSED_SET' or 'SPLIT_OPEN_SET'
SPLIT_COLUMN = 'SPLIT_CLOSED_SET' 

def prepare_yolo_data():
    # 1. Load Data
    bbox_df = pd.read_csv(BBOX_CSV)
    split_df = pd.read_csv(SPLIT_CSV)
    
    # 2. Merge dataframes on VIDEO_ID
    # We dynamically select the column based on SPLIT_COLUMN
    df = pd.merge(bbox_df, split_df[['VIDEO_ID', SPLIT_COLUMN]], on='VIDEO_ID')
    
    # Filter for 'train' only in the selected split mode
    train_df = df[df[SPLIT_COLUMN] == 'train'].copy()
    
    print(f"📊 Mode: {SPLIT_COLUMN}")
    print(f"📊 Found {len(train_df)} total training boxes.")

    # 3. Create YOLO structure
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

    # Internal split (90% train, 10% val) for YOLO's own training process
    unique_vids = train_df['VIDEO_ID'].unique()
    if len(unique_vids) == 0:
        print(f"❌ ERROR: No training data found for {SPLIT_COLUMN}. Check your CSV values.")
        return False
        
    split_idx = int(len(unique_vids) * 0.9)
    yolo_train_vids = unique_vids[:split_idx]

    print("🚀 Processing images...")

    for _, row in train_df.iterrows():
        dog_id = str(row['DOG_ID'])
        video_id = str(row['VIDEO_ID'])
        subset = 'train' if video_id in yolo_train_vids else 'val'
        
        img_name = f"{dog_id}-{video_id}.jpg"
        src_path = os.path.join(IMAGES_ROOT, dog_id, img_name)
        
        if not os.path.exists(src_path):
            src_path = src_path.replace('.jpg', '.jpeg')
            if not os.path.exists(src_path):
                continue

        # 4. Get size and calculate coordinates
        with Image.open(src_path) as img:
            img_w, img_h = img.size

        x_center = (row['x_top_left'] + (row['width'] / 2)) / img_w
        y_center = (row['y_top_left'] + (row['height'] / 2)) / img_h
        w_norm = row['width'] / img_w
        h_norm = row['height'] / img_h

        # Clamp for YOLO safety
        x_center = max(0.001, min(0.999, x_center))
        y_center = max(0.001, min(0.999, y_center))
        w_norm = max(0.001, min(0.999, w_norm))
        h_norm = max(0.001, min(0.999, h_norm))

        # 5. Write Label
        label_name = img_name.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(OUTPUT_DIR, 'labels', subset, label_name)
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 6. Copy Image
        dst_path = os.path.join(OUTPUT_DIR, 'images', subset, img_name)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
    
    return True

def train_custom_dog_detector():
    # Dynamically name the run based on the split mode
    run_name = f"dog_detector_{SPLIT_COLUMN.lower()}"
    
    model = YOLO('yolov8n.pt')
    model.train(
        data='dog_data.yaml',
        epochs=50,
        imgsz=640, 
        batch=16,
        name=run_name,
        device=0 
    )
    print(f"\n🚀 Training complete! Model saved at runs/detect/{run_name}/weights/best.pt")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    if prepare_yolo_data():
        # Create YAML
        yaml_path = os.path.join(os.getcwd(), 'dog_data.yaml')
        yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
names:
  0: dog
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        train_custom_dog_detector()