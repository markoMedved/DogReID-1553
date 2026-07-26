import torch
from pathlib import Path

class Config:
    """
    Dog Re-ID training configuration.
    """
    
    # --- Experiment Settings ---
    backbone      = "dinov2"      # Options: 'dinov2', 'vit', 'swin', 'convnext'
    reid_method   = "bot"         # Options: 'bot', 'transreid'
    world         = "closed"      # Options: 'closed', 'open'
    pooling_type  = "attention"   # Options: 'attention', 'mean', 'max'
    full_finetune = True

    # --- Image & Model Architecture ---
    img_size      = (252, 182)    # 252x182 for DINOv2 (patch 14), 256x192 for patch-16 backbones
    jpm_parts     = 4             # k local parts for TransReID JPM
    embedding_dim = 768           # Default embedding dimension

    # --- Directory Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"

    # --- Hardware & Compute ---
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0               # Number of dataloader workers
    chunk_size  = 16              # Number of frames processed simultaneously

    # --- Batch Sampling (PK Strategy) ---
    batch_size = 16              # Total batch size (P identities × K clips)
    k          = 4               # Number of clips per identity
    num_ids    = batch_size // k 
    clip_len   = 16             # Frame length of each video clip
    val_split  = 0               # Validation set ratio
    
    # --- Training & Optimization ---
    epochs        = 80
    warmup_epochs = 10
    lr_milestones = (40, 70)
    lr_gamma      = 0.1
    weight_decay  = 1e-05
    margin        = 0.3          # Margin for triplet loss
    id_loss_weight = 1.0         # Weight for identity loss
    lr            = 2e-05        # Learning rate
    accum_steps   = 8            # Gradient accumulation steps

    # --- Data Augmentation ---
    aug_pad = 10                  # Padding for random crop
    re_prob = 0.5                 # Random erasing probability

    # --- Evaluation ---
    eval_period = 100            # Epochs between evaluations 
    eval_only   = False  

    def __init__(self):
        """Dynamic configuration initialization."""
        # 1. Backwards compatibility for legacy 'model' attribute
        self.model = self.backbone

        # 2. Dynamic output run naming (per Section 3.1)
        self.run_name = (
            f"{self.backbone}_{self.reid_method}_{self.world}_"
            f"{self.pooling_type}_finetune_{self.full_finetune}"
        )
        self.output_dir = self.project_root / "trained_models" / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Model-specific embedding dimensions
        if self.backbone == "swin":
            self.embedding_dim = 1024

    def display(self):
        """Print a formatted table of the configuration settings."""
        print("\n" + "="*50)
        print(f"DOG RE-ID CONFIGURATION: {self.run_name}")
        print("-"*50)
        
        sections = {
            "DATA": ["world", "batch_size", "k", "clip_len", "img_size", "num_workers"],
            "MODEL": ["backbone", "reid_method", "pooling_type", "full_finetune", "jpm_parts", "embedding_dim", "chunk_size"],
            "OPTIM": ["lr", "epochs", "warmup_epochs", "lr_milestones", "accum_steps", "margin", "weight_decay"],
            "AUGMENTATION": ["aug_pad", "re_prob"],
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
        return f"<Config: {self.run_name} | Method: {self.reid_method} | Backbone: {self.backbone} | Device: {self.device}>"