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

    # --- Backbone Freezing ---
    # Defaults match the original experiments: the last two transformer blocks
    # plus the final norm. full_finetune=True trains everything and ignores
    # unfreeze_blocks. unfreeze_blocks=0 freezes the whole backbone.
    full_finetune   = False
    unfreeze_blocks = 2

    # --- Image & Model Architecture ---
    # 224x224 matches the published experiments and keeps the new runs
    # comparable to them on this axis. The dog boxes in bounding_boxes.csv have
    # a geometric-mean width:height of 0.81, so a non-square input fits the data
    # better; (252, 182) is the DINOv2 patch-14 equivalent and is reported as a
    # separate ablation rather than changed by default.
    img_size        = (224, 224)
    dinov2_variant  = "vitb14_reg"

    # TransReID Jigsaw Patch Module
    jpm_parts          = 4        # local branches
    jpm_shift          = 5        # token roll before shuffling
    jpm_shuffle_groups = 2        # interleaved groups

    # --- Directory Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"

    # --- Hardware & Compute ---
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12              # Dataloader workers; 0 serializes decoding with compute
    chunk_size  = 16              # Number of frames processed simultaneously
    amp         = "bf16"          # 'bf16', 'fp16' or None. CUDA only.

    # --- Batch Sampling (PK Strategy) ---
    # Hard-negative mining operates within a single forward pass, so the number
    # of identities per batch (batch_size // k) matters more than the effective
    # batch size reached through accumulation.
    # With unfreeze_blocks=2 the frozen leading blocks run under no_grad, so
    # activation memory is a fraction of a full fine-tune and the batch can be
    # far larger. Raise batch_size until it no longer fits; more identities per
    # batch is the single biggest lever on triplet mining quality.
    batch_size = 32              # Total batch size (P identities × K clips)
    k          = 4               # Clips per identity -> 8 identities per batch
    clip_len   = 8               # Frames per video clip
    val_split  = 0               # Validation set ratio

    # --- Training & Optimization ---
    # Milestones are epoch indices and must fit inside `epochs`; rescale both
    # together or the learning rate never decays.
    epochs        = 40
    warmup_epochs = 5
    lr_milestones = (25, 35)
    lr_gamma      = 0.1
    weight_decay  = 1e-05
    margin        = 0.3          # Margin for triplet loss
    id_loss_weight = 1.0         # Weight for identity loss
    lr            = 2e-05        # Learning rate
    warmup_factor = 0.01         # Starting fraction of the base LR
    accum_steps   = 2            # Gradient accumulation steps
    save_period   = 10           # Epochs between intermediate checkpoints

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

        # 2. Derived sampling quantity
        self.num_ids = self.batch_size // self.k

        # 3. Dynamic output run naming
        self.refresh_run_name()

        # 4. Model-specific embedding dimensions
        if self.backbone == "swin":
            self.embedding_dim = 1024

    @staticmethod
    def compose_run_name(backbone, reid_method, world, pooling_type, full_finetune):
        """Single definition of the run name, shared by training and evaluation."""
        return f"{backbone}_{reid_method}_{world}_{pooling_type}_finetune_{full_finetune}"

    def refresh_run_name(self, make_dir=True):
        """Recompute run_name and output_dir after any field is overridden."""
        self.num_ids = self.batch_size // self.k
        self.run_name = self.compose_run_name(
            self.backbone, self.reid_method, self.world,
            self.pooling_type, self.full_finetune,
        )
        self.output_dir = self.project_root / "trained_models" / self.run_name
        if make_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.run_name

    def display(self):
        """Print a formatted table of the configuration settings."""
        print("\n" + "="*50)
        print(f"DOG RE-ID CONFIGURATION: {self.run_name}")
        print("-"*50)
        
        sections = {
            "DATA": ["world", "batch_size", "k", "num_ids", "clip_len", "img_size", "num_workers"],
            "MODEL": ["backbone", "reid_method", "pooling_type", "full_finetune", "unfreeze_blocks", "jpm_parts", "chunk_size"],
            "OPTIM": ["lr", "epochs", "warmup_epochs", "lr_milestones", "accum_steps", "margin", "weight_decay", "id_loss_weight", "amp"],
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