import sys
import os
import torch
from pathlib import Path
from collections import OrderedDict

# =================================================================
# 🛠️ CONFIGURABLE SETTINGS (CHANGE THESE)
# =================================================================
WORLD_TYPE = "closed"          # Options: "closed" or "open"
MODEL_NAME = "dinov2"
MODEL_PATH = f"/d/hpc/projects/FRI/mm12755/DogReID-1553/DogReID-1553/experiments/{MODEL_NAME}_{WORLD_TYPE}_v1/best_model.pth"
OUTPUT_FOLDER = f"csvs/dinov2_{WORLD_TYPE}_v1"
CSV_NAME = f"{WORLD_TYPE}_dist_matrix.csv"
# =================================================================

# 1. System Path Setup
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.dinov2_builder import DINOv2ReID 
from data.dataloader import build_test_loaders
from configs.config import Config
from evaluation_utils import (
    generate_distance_csv,
    calculate_metrics_from_csv,
    calculate_open_set_metrics_from_csv
)

# 2. Setup Configuration
cfg = Config()
cfg.world = WORLD_TYPE
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.output_dir = Path(OUTPUT_FOLDER)
cfg.output_dir.mkdir(parents=True, exist_ok=True)

# 3. Initialize and Load Model
print(f"-> Loading Model Architecture...")
model = DINOv2ReID() 

print(f"-> Loading Weights: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)
state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

# Clean 'module.' prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(cfg.device)
model.eval()

# 4. Build Dataloaders (Uses cfg.world set above)
print(f"-> Preparing {cfg.world.upper()} test dataloaders...")
query_loader, gallery_loader = build_test_loaders(cfg)

# 5. Generate and Save Distance CSV
print(f"-> Running Inference...")
csv_path = generate_distance_csv(
    model, 
    query_loader, 
    gallery_loader, 
    cfg, 
    filename=CSV_NAME
)

# 6. Run Metrics based on World Type
print("\n" + "="*30)
if cfg.world == "closed":
    print("   CLOSED WORLD RESULTS")
    calculate_metrics_from_csv(csv_path)
else:
    print("   OPEN SET RESULTS")
    calculate_open_set_metrics_from_csv(csv_path)