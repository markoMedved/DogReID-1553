import torch
from pathlib import Path

class Config:
    # --- Paths ---
    # Using .resolve() is great. Ensure data_root actually points to the 
    # folder containing "Videos" and "splits.csv"
    data_root = Path(__file__).resolve().parent.parent
    split_file = data_root / "splits.csv"

    output_dir = "trained_models_vit" # Change if change model
    model = "vit" # Change if change model

    world = "closed"

    batch_size = 8

    num_ids = 4
    num_instances = 2
    
    num_workers = 4
    clip_len = 16 

    embedding_dim = 768 

    epochs = 50
    lr = 3e-4
    weight_decay = 1e-5
    eval_period = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Evaluation / Resume ----
    eval_only = False 
    checkpoint_path = output_dir / "best_model.pth"

    def __init__(self):
        # Create output directory automatically
        self.output_dir.mkdir(parents=True, exist_ok=True)