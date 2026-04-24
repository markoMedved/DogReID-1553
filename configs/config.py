import torch
from pathlib import Path

class Config:
    """
    Dog Re-ID training configuration.
    Designed for DINOv2 / Swin on H100 (MIG 32GB).
    """
    
    # --- Experiment ---
    model = "vit"   # 'dinov2', 'swin', 'vit'
    world = "closed"   # 'closed' or 'open'
    run_name = f"{model}_{world}_v1"

    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"
    
    # outputs
    output_dir   = project_root / "experiments" / run_name
    checkpoint_path = output_dir / "best_model.pth"

    # --- Hardware ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12      # dataloader workers
    chunk_size  = 16      # frames processed at once

    # --- Batch Sampling (PK) ---
    # batch = P identities × K clips
    batch_size = 16     
    k = 4                
    num_ids = batch_size // k 
    
    # video clip length
    clip_len = 16    

    val_split = 0

    # --- Model ---
    embedding_dim = 768
    
    # --- Optimization ---
    epochs = 50
    weight_decay = 5e-05
    margin = 0.3
    lr = 1e-05
    
    # gradient accumulation (simulate larger batch)
    accum_steps = 8      

    # --- Evaluation ---
    eval_period = 100
    eval_only   = False

    def __init__(self):
            """Create experiment directory."""
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.model == "swin":
                self.embedding_dim = 1024
            

    def display(self):
        """Print config table."""
        print("\n" + "="*50)
        print(f"DOG RE-ID CONFIGURATION: {self.run_name}")
        print("-"*50)
        
        sections = {
            "DATA": ["world", "batch_size", "k", "clip_len", "num_workers"],
            "MODEL": ["model", "embedding_dim", "chunk_size"],
            "OPTIM": ["lr", "epochs", "accum_steps", "margin", "weight_decay"],
            "PATHS": ["output_dir"]
        }

        for section, keys in sections.items():
            print(f"[{section}]")
            for key in keys:
                val = getattr(self, key)
                if isinstance(val, Path):
                    val = f".../{val.name}"
                print(f"  {key:<15} : {val}")
        
        print("="*50 + "\n")

    def __repr__(self):
        """Short config summary."""
        return f"<Config: {self.run_name} | Model: {self.model} | Device: {self.device}>"