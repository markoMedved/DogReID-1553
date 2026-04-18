import torch
from pathlib import Path

class Config:
    """
    Configuration for Dog Re-ID Training.
    Optimized for DINOv2/Swin Backbones on H100 (MIG 32GB).
    """
    
    # --- Experimental Metadata ---
    model = "dinov2"      # Options: 'dinov2', 'swin', 'vit'
    world = "closed"      # Options: 'closed' (fixed dog set), 'open' (new dogs at test)
    run_name = f"{model}_{world}_v1"

    # --- Path Management ---
    # Automatically finds project root based on this file's location
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"
    
    # Checkpoints and logs stored in a model-specific folder
    output_dir   = project_root / "experiments" / run_name
    checkpoint_path = output_dir / "best_model.pth"

    # --- Hardware & Performance ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8      # Parallel data loading (CPU)
    chunk_size  = 16    # Max frames processed by GPU simultaneously 

    # --- Data Sampling (The PK Strategy) ---
    # batch_size = P * K
    # P = Number of unique dog IDs in a batch
    # K = Number of clips per dog ID
    batch_size = 16      
    k = 4                
    num_ids = batch_size // k 
    
    # Video specific: number of frames per clip
    clip_len = 16         

    # --- Model Hyperparameters ---
    # Note: DINOv2-Base and vit is 768, Swin-Base is 1024
    embedding_dim = 768
    
    # --- Optimization ---
    epochs = 50
    lr = 3e-5            # Gentle start for fine-tuning foundation models
    weight_decay = 1e-4  # L2 penalty to prevent overfitting on 3.5k samples
    margin = 0.3         # Minimum distance gap for Triplet Loss
    
    # Gradient Accumulation: Simulates a larger batch size (16 * 8 = 128)
    # This leads to much smoother loss curves and better convergence.
    accum_steps = 8      

    # --- Evaluation ---
    eval_period = 1      # Run gallery/query validation every N epochs
    eval_only   = False  # Set True to skip training and just run test

    def __init__(self):
            """Initializes the experiment directory."""
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Use our new display method during init
            self.display()

    def display(self):
        """Prints a structured table of the current configuration."""
        print("\n" + "="*50)
        print(f"🐾 DOG RE-ID CONFIGURATION: {self.run_name}")
        print("-"*50)
        
        # Group important settings for readability
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
                # Shorten paths so they don't break the formatting
                if isinstance(val, Path):
                    val = f".../{val.name}"
                print(f"  {key:<15} : {val}")
        
        print("="*50 + "\n")

    def __repr__(self):
        """Clean summary for logging."""
        return f"<Config: {self.run_name} | Model: {self.model} | Device: {self.device}>"