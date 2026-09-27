import os
import torch
from pathlib import Path


class Config:
    """
    Dog Re-ID training configuration.
    """
    
    # --- Experiment Settings ---
    model         = "dinov2"       # Options: 'dinov2', 'swin', 'vit', 'convnetxt'
    backbone      = "dinov2"       # Options: 'dinov2', 'osnet', 'megadescriptor', 'vit', 'swin', 'convnext', 'miewid'
    reid_method   = None           # Options: None (baseline/rebuttal), 'bot', 'transreid'
    world         = "closed"       # Options: 'closed', 'open'
    pooling_type  = "attention"    # Options: 'attention', 'mean', 'max'
    full_finetune = False
    use_id_loss   = True           # Identity Classification Loss for baselines
    unfreeze_blocks = 2            # Trailing backbone blocks to train when not full fine-tuning

    # --- Image & Model Architecture ---
    img_size        = (224, 224)
    embedding_dim   = 768
    num_classes     = 0            # Populated dynamically in train.py from dataset
    dinov2_variant  = "vitb14_reg"
    osnet_variant    = "osnet_ain_x1_0"
    osnet_pretrained = True
    osnet_weights    = None
    megadescriptor_variant = "hf-hub:BVRA/MegaDescriptor-L-224"

    # TransReID Jigsaw Patch Module
    jpm_parts          = 4
    jpm_shift          = 5
    jpm_shuffle_groups = 2

    # --- Directory Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root 
    split_file   = project_root / "splits.csv"

    # --- Hardware & Compute ---
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 12
    chunk_size  = 16
    amp         = "bf16"           # 'bf16', 'fp16' or None. CUDA only.

    # --- Batch Sampling (PK Strategy) ---
    batch_size = 16
    k          = 4
    num_ids    = batch_size // k
    clip_len   = 16                # Default 16 for baseline; 8 used in BoT experiments
    val_split  = 0.2

    # --- Training & Optimization ---
    epochs         = 100           # 100 for baseline, 51 for bot
    warmup_epochs  = 5
    lr_milestones  = (25, 35)
    lr_gamma       = 0.1
    weight_decay   = 1e-05
    margin         = 0.3
    id_loss_weight = 1.0
    lr             = 5e-05         # 5e-05 for baseline, 2e-05 for bot
    warmup_factor  = 0.01
    accum_steps    = 8             # 8 for baseline, 2 for bot
    save_period    = 10

    # --- Data Augmentation ---
    aug_pad = 10
    re_prob = 0.5

    # --- Evaluation ---
    eval_period = 1
    eval_only   = False  

    # Experiment for background noise
    mask_dog = False 
    bbox_file = project_root / "bounding_boxes.csv"
    use_gt_for_query_mask = False
    use_gt_for_gallery_mask = False
    force_yolo = True

    def __init__(self):
        self.update_model_settings()

    def update_model_settings(self):
        """
        Call this after updating self.model / self.backbone via CLI args.
        Adjusts dimensions, image sizes, and output directories.
        """
        if hasattr(self, "model") and self.model:
            self.model = self.model.lower()
            self.backbone = self.model
        elif hasattr(self, "backbone") and self.backbone:
            self.backbone = self.backbone.lower()
            self.model = self.backbone

        self.num_ids = self.batch_size // self.k

        if "swin" in self.backbone:
            self.embedding_dim = 1024
            self.img_size = (192, 192)
        elif self.backbone.startswith("oa_") or self.backbone.startswith("openanimals_"):
            self.embedding_dim = 2048 * (8 if "mgn" in self.backbone else 1)
            self.img_size = (256, 256) if "bot" in self.backbone else (384, 384)
            if any(k in self.backbone for k in ("arbase", "mgn", "sbs")):
                self.re_prob = 0.0
        else:
            self.embedding_dim = 768
            self.img_size = (224, 224)

        self.refresh_run_name()

    @staticmethod
    def compose_run_name(backbone, reid_method, world, pooling_type, full_finetune, use_id_loss=None, mask_dog=False):
        """Single definition of the run name, shared by training and evaluation."""
        if use_id_loss is None:
            use_id_loss = getattr(Config, "use_id_loss", True)
        if reid_method in ("bot", "transreid"):
            name = f"{backbone}_{reid_method}_{world}_{pooling_type}_finetune_{full_finetune}"
        else:
            name = f"{backbone}_{world}_{pooling_type}_finetune_{full_finetune}_idloss_{use_id_loss}"
        if mask_dog:
            name += "_masked"
        return name

    def refresh_run_name(self, make_dir=True):
        """Recompute run_name and output_dir after any field is overridden."""
        self.num_ids = self.batch_size // self.k
        self.run_name = self.compose_run_name(
            self.backbone, self.reid_method, self.world,
            self.pooling_type, self.full_finetune, getattr(self, "use_id_loss", False),
            getattr(self, "mask_dog", False)
        )
        self.output_dir = self.project_root / "trained_models" / self.run_name
        if make_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.run_name

    def display(self):
        """Print a formatted table of the configuration settings."""
        print("\n" + "="*50)
        print(f"DOG RE-ID CONFIGURATION: {getattr(self, 'run_name', self.model)}")
        print("-"*50)
        
        sections = {
            "DATA": ["world", "batch_size", "k", "num_ids", "clip_len", "img_size", "num_workers"],
            "MODEL": ["backbone", "reid_method", "model", "embedding_dim", "num_classes", "pooling_type", "full_finetune", "unfreeze_blocks", "use_id_loss"],
            "OPTIM": ["lr", "epochs", "warmup_epochs", "lr_milestones", "accum_steps", "margin", "weight_decay", "id_loss_weight", "amp"],
            "AUGMENTATION": ["aug_pad", "re_prob"],
            "PATHS": ["output_dir"]
        }

        for section, keys in sections.items():
            print(f"[{section}]")
            for key in keys:
                val = getattr(self, key, None)
                if isinstance(val, Path):
                    val = f".../{val.name}"
                print(f"  {key:<15} : {val}")
        
        if self.device.type == "cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS"):
                print("  [CRITICAL WARNING] SLURM allocated a GPU, but PyTorch cannot use CUDA on this node!")
                print("                     Check 'nvidia-smi' or hardware status for errors.")
            else:
                print("  [NOTE] Running on CPU.")

        print("="*50 + "\n")

    def __repr__(self):
        """Return a short summary of the config instance."""
        if getattr(self, "reid_method", None):
            return f"<Config: {self.run_name} | Method: {self.reid_method} | Backbone: {self.backbone} | Device: {self.device}>"
        return f"<Config: {getattr(self, 'run_name', self.model)} | Device: {self.device}>"
