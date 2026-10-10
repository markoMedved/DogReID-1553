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
    batch_size = 64
    k          = 4
    num_ids    = batch_size // k
    clip_len   = 8               
    val_split  = 0.2

    # --- Training & Optimization ---
    epochs         = 50            # 50 epochs total
    warmup_epochs  = 10            # 10‑epoch linear warmup (3.5e‑5 → 3.5e‑4)
    lr_milestones  = (15, 30)      # decay LR by 0.1 at epoch 15 and 30
    lr_gamma       = 0.1
    optimizer      = "adam"        # 'adam', 'adamw', or 'sgd'
    weight_decay   = 5e-04         # 5e-4 matches OpenAnimals across all baselines
    margin         = 0.3          # Triplet loss margin α
    id_loss_weight = 1.0
    center_loss_weight = 5e-04   # optional center loss β
    lr             = 3.5e-04      # base LR (OpenAnimals uses 3.5e-4 for both backbone and heads)
    backbone_lr_factor = 1.0      # 1.0 matches OpenAnimals (backbone LR == head LR == base LR)
    warmup_factor  = 0.1          # 0.1 matches OpenAnimals (linear warmup from 0.1x to 1.0x)
    accum_steps    = 1            # gradient accumulation (1 for standard paper baseline)
    save_period    = 0            # 0 disables periodic checkpoints (saves only latest_model.pth)
    backbone_drop_epoch = None    # Epoch at which backbone LR drops (e.g. 5)
    backbone_drop_factor = 0.1    # Multiplier for backbone LR after drop epoch

    # --- Data Augmentation ---
    aug_pad = 10
    re_prob = 0.5
    autoaug_prob = 0.0

    # --- Evaluation ---
    eval_period = 1
    eval_only   = False  

    # Experiment for background noise
    mask_dog = False 
    bbox_file = project_root / "bounding_boxes.csv"
    use_gt_for_query_mask = False
    use_gt_for_gallery_mask = False
    force_yolo = False

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

        native_or_oa_baselines = (
            "bot", "oa_bot", "openanimals_bot",
            "oa_dinov2", "oa_dinov2_bot", "openanimals_dinov2_bot",
            "agw", "oa_agw", "openanimals_agw",
            "sbs", "oa_sbs", "openanimals_sbs",
            "mgn", "oa_mgn", "openanimals_mgn",
            "arbase", "oa_arbase", "openanimals_arbase",
            "arbase_mb", "oa_arbase_mb", "arbase_mgn",
        )

        if "swin" in self.backbone:
            self.embedding_dim = 1024
            self.img_size = (192, 192)
        elif self.backbone in native_or_oa_baselines or self.backbone.startswith("oa_") or self.backbone.startswith("openanimals_"):
            is_mb = "mgn" in self.backbone or "arbase_mb" in self.backbone
            self.embedding_dim = 768 if "dinov2" in self.backbone else (2048 * (8 if is_mb else 1))
            
            # DogReID settings (Square aspect ratio, Adam, WD=5e-4, 1.0x BB LR for full, 0.1x for frozen):
            self.optimizer = "adam"
            self.weight_decay = 5e-04
            self.backbone_lr_factor = 1.0 if getattr(self, "full_finetune", False) else 0.1
            self.chunk_size = 64
            self.autoaug_prob = 0.1 if "sbs" in self.backbone else 0.0  # Base-SBS: AUTOAUG p=0.1

            if "dinov2" in self.backbone:
                self.img_size = (224, 224)
                self.re_prob = 0.5
                self.margin = 0.3
                self.lr = 3.5e-04
                self.chunk_size = 32
                if not getattr(self, "full_finetune", False):
                    self.backbone_lr_factor = 0.1
            elif "mgn" in self.backbone:
                self.img_size = (384, 128)
                self.re_prob = 0.0          # REA disabled for MGN
                self.margin = 0.3
                self.epochs = 50
                self.lr = 3.5e-04
                self.lr_sched = "cosine"
                self.lr_delay_epochs = 12
                self.chunk_size = 64
                self.backbone_drop_epoch = 5
            elif "arbase" in self.backbone:
                self.img_size = (384, 384)
                self.re_prob = 0.0          # REA disabled for ARBase
                self.margin = 0.3
                self.epochs = 50
                self.lr = 3.5e-04
                self.lr_sched = "cosine"
                self.lr_delay_epochs = 12
                self.chunk_size = 64
                self.backbone_drop_epoch = 5
            elif "agw" in self.backbone:
                self.img_size = (256, 128)
                self.re_prob = 0.5          # REA enabled for AGW
                self.margin = 0.0           # Weighted soft-margin triplet
                self.epochs = 50
                self.lr = 3.5e-04
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 30)
                self.chunk_size = 64
                self.backbone_drop_epoch = 5
            elif "sbs" in self.backbone:
                self.img_size = (384, 128)
                self.re_prob = 0.5          # REA enabled for SBS
                self.autoaug_prob = 0.1     # AutoAugment p=0.1
                self.margin = 0.0           # Soft-margin triplet
                self.epochs = 50
                self.lr = 3.5e-04
                self.lr_sched = "cosine"
                self.lr_delay_epochs = 12
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            elif "bot" in self.backbone:
                self.img_size = (256, 128)
                self.re_prob = 0.5
                self.margin = 0.3
                self.epochs = 120
                self.lr = 3.5e-04
                self.lr_sched = "multistep"
                self.lr_milestones = (40, 90)
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            else:
                self.img_size = (256, 128)
                self.re_prob = 0.5
                self.chunk_size = 64

            if getattr(self, "reid_method", None) is None:
                self.reid_method = "bot"
        elif self.backbone in ("psta", "video_psta"):
            self.embedding_dim = 1024
            self.img_size = (256, 128)
            self.clip_len = 8
            self.re_prob = 0.5
            self.chunk_size = 64
            self.optimizer = "adam"
            self.weight_decay = 5e-04
            self.backbone_lr_factor = 1.0
            self.lr = 3.5e-04
            self.margin = 0.3
            self.epochs = 500
            self.warmup_epochs = 10
            self.warmup_factor = 0.01
            self.lr_sched = "multistep"
            self.lr_milestones = (70, 140, 210, 310, 410)
            self.lr_gamma = 0.3
            self.backbone_drop_epoch = None
        elif "resnet" in self.backbone:
            self.embedding_dim = 2048
            self.img_size = (256, 128)
            self.re_prob = 0.5
            self.chunk_size = 64
            self.optimizer = "adam"
            self.weight_decay = 5e-04
            self.backbone_lr_factor = 1.0 if getattr(self, "full_finetune", False) else 0.1
            self.lr = 3.5e-04
            self.margin = 0.3
            self.epochs = 50
            self.warmup_epochs = 10
            self.lr_sched = "multistep"
            self.lr_milestones = (15, 30)
            self.backbone_drop_epoch = 5
        elif "dinov2" in self.backbone:
            self.embedding_dim = 768
            self.img_size = (224, 224)
            self.re_prob = 0.5
            self.chunk_size = 32
            self.optimizer = "adam"
            self.weight_decay = 5e-04
            self.backbone_lr_factor = 1.0 if getattr(self, "full_finetune", False) else 0.1
            self.lr = 3.5e-04
            self.margin = 0.3
            self.epochs = 50
            self.warmup_epochs = 10
            self.lr_sched = "multistep"
            self.lr_milestones = (15, 30)
        else:
            self.embedding_dim = 768
            self.img_size = (224, 224)

        self.refresh_run_name(make_dir=False)

    @staticmethod
    def compose_legacy_run_name(backbone, reid_method, world, pooling_type, full_finetune, use_id_loss=None, mask_dog=False):
        """Old verbose naming definition preserved for checkpoint fallback."""
        if use_id_loss is None:
            use_id_loss = getattr(Config, "use_id_loss", True)
        if reid_method in ("bot", "transreid"):
            name = f"{backbone}_{reid_method}_{world}_{pooling_type}_finetune_{full_finetune}"
        else:
            name = f"{backbone}_{world}_{pooling_type}_finetune_{full_finetune}_idloss_{use_id_loss}"
        if mask_dog:
            name += "_masked"
        return name

    @staticmethod
    def compose_run_name(backbone, reid_method=None, world="closed", pooling_type="attention", full_finetune=False, use_id_loss=None, mask_dog=False, backbone_lr_factor=None, backbone_drop_epoch=None):
        """Clean and concise run naming convention."""
        bb = str(backbone).lower() if backbone else ""
        method = str(reid_method).lower() if reid_method else ""
        if bb in ("arbase", "agw", "sbs", "mgn", "psta", "video_psta"):
            base = "psta" if "psta" in bb else bb
            if backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 1.0) > 1e-4:
                tune = f"{float(backbone_lr_factor):g}x"
                name = f"{base}_{tune}"
            else:
                name = base
        elif "dinov2" in bb:
            base = "dinov2_bot" if (method in ("bot", "transreid") or "bot" in bb) else "dinov2"
            if full_finetune:
                name = f"{base}_full"
            elif backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 0.1) > 1e-4:
                name = f"{base}_{float(backbone_lr_factor):g}x"
            else:
                name = base
        elif "resnet" in bb:
            base = "resnet50_bot" if (method in ("bot", "transreid") or "bot" in bb) else "resnet50"
            if not full_finetune:
                name = f"{base}_frozen"
            elif backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 1.0) > 1e-4:
                name = f"{base}_{float(backbone_lr_factor):g}x"
            else:
                name = base
        else:
            if method in ("bot", "transreid") and not bb.endswith(method):
                base = f"{bb}_{method}"
            else:
                base = bb
            if full_finetune:
                tune = f"unfrozen_{float(backbone_lr_factor):g}x" if (backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 1.0) > 1e-4) else "full"
            else:
                tune = f"frozen_{float(backbone_lr_factor):g}x" if (backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 0.1) > 1e-4) else "frozen"
            name = f"{base}_{tune}"

        if backbone_drop_epoch is not None:
            name += f"_drop{backbone_drop_epoch}"

        if world and str(world).lower() != "closed":
            name += f"_{world}"
        if pooling_type and str(pooling_type).lower() != "attention":
            name += f"_{pooling_type}"
        if mask_dog:
            name += "_masked"
        return name

    def refresh_run_name(self, make_dir=False):
        """Recompute run_name and output_dir after any field is overridden."""
        self.num_ids = self.batch_size // self.k
        self.run_name = self.compose_run_name(
            self.backbone, self.reid_method, self.world,
            self.pooling_type, self.full_finetune, getattr(self, "use_id_loss", False),
            getattr(self, "mask_dog", False), getattr(self, "backbone_lr_factor", None),
            getattr(self, "backbone_drop_epoch", None)
        )
        # Only map to legacy name if using the original regime backbone_lr_factor (1.0 for full, 0.1 for frozen) and no backbone_drop_epoch
        bb_factor = getattr(self, "backbone_lr_factor", 1.0 if self.full_finetune else 0.1)
        expected_bb_factor = 1.0 if self.full_finetune else 0.1
        if getattr(self, "backbone_drop_epoch", None) is not None:
            self.legacy_run_name = None
            self.legacy_output_dir = None
        elif abs(float(bb_factor) - expected_bb_factor) < 1e-4:
            self.legacy_run_name = self.compose_legacy_run_name(
                self.backbone, self.reid_method, self.world,
                self.pooling_type, self.full_finetune, getattr(self, "use_id_loss", False),
                getattr(self, "mask_dog", False)
            )
            self.legacy_output_dir = self.project_root / "checkpoints" / self.legacy_run_name
        else:
            self.legacy_run_name = None
            self.legacy_output_dir = None
        self.output_dir = self.project_root / "checkpoints" / self.run_name
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
