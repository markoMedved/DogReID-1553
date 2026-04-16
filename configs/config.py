import torch
from pathlib import Path

class Config:
    # --- Paths ---
    data_root = Path(__file__).resolve().parent.parent
    split_file = data_root / "splits.csv"

    # FIX: Wrap the string in Path() so the / operator works below
    output_dir = Path("trained_models_vit") 
    
    model = "vit"
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
    
    # Now that output_dir is a Path object, this line will work perfectly
    checkpoint_path = output_dir / "best_model.pth"

    def __init__(self):
        # Ensure the directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)