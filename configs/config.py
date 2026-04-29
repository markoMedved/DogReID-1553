import torch
from pathlib import Path

class Config:
    """
    Dog Re-ID training configuration.
    """
    
    # --- Experiment Settings ---
    model = "swin"       # Options: 'dinov2', 'swin', 'vit'
    world = "closed"       # Options: 'closed', 'open'
    run_name = f"{model}_{world}"

    # --- Directory Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"
    output_dir   = project_root / "trained_models" / f"{model}_{world}"

    # --- Hardware & Compute ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12    # Number of dataloader workers
    chunk_size  = 16     # Number of frames processed simultaneously

    # --- Batch Sampling (PK Strategy) ---
    batch_size = 16      # Total batch size (P identities × K clips)
    k = 4                # Number of clips per identity
    num_ids = batch_size // k 
    clip_len = 16        # Frame length of each video clip
    val_split = 0     # Validation set ratio

    # --- Model Architecture ---
    embedding_dim = 768  # Default embedding dimension
    
    # --- Training & Optimization ---
    epochs = 50
    weight_decay = 1e-05
    margin = 0.3       # Loss margin
    lr = 2e-05           # Learning rate
    accum_steps = 8      # Gradient accumulation steps to simulate larger batch

    # --- Evaluation ---
    eval_period = 100     # Epochs between evaluations 
    eval_only   = False  

    def __init__(self):
        """Create experiment directory and apply model-specific overrides."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model == "swin":
            self.embedding_dim = 1024

    def display(self):
        """Print a formatted table of the configuration settings."""
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
        """Return a short summary of the config instance."""
        return f"<Config: {self.run_name} | Model: {self.model} | Device: {self.device}>"