import os
import torch
from pathlib import Path


class Config:
    """
    Dog Re-ID training configuration.
    """
    
    # --- Experiment Settings ---
    model         = "dinov2"       # Options: 'dinov2', 'tfclip', 'swin', 'vit', 'convnetxt'
    backbone      = "dinov2"       # Options: 'dinov2', 'tfclip', 'osnet', 'megadescriptor', 'vit', 'swin', 'convnext', 'miewid'
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
    eval_period = 5
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

        native_or_oa_baselines = (
            "bot", "oa_bot", "openanimals_bot",
            "dinov2_bot", "oa_dinov2", "oa_dinov2_bot", "openanimals_dinov2_bot",
            "agw", "oa_agw", "openanimals_agw",
            "dinov2_agw", "oa_dinov2_agw", "openanimals_dinov2_agw",
            "sbs", "oa_sbs", "openanimals_sbs",
            "dinov2_sbs", "oa_dinov2_sbs", "openanimals_dinov2_sbs",
            "mgn", "oa_mgn", "openanimals_mgn",
            "arbase", "oa_arbase", "openanimals_arbase",
            "dinov2_arbase", "oa_dinov2_arbase", "openanimals_dinov2_arbase",
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
                self.epochs = 50
                self.eval_period = 5
                self.lr = 3.5e-04
                self.backbone_lr_factor = 0.1
                self.optimizer = "adam"
                self.weight_decay = 5e-04
                self.amp = "bf16"
                self.batch_size = 64
                self.k = 4
                self.accum_steps = 1
                self.num_workers = 12
                self.clip_len = 8
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.unfreeze_blocks = getattr(self, "unfreeze_blocks", 2)

                if "sbs" in self.backbone:
                    self.freeze_epochs = 5
                    self.autoaug_prob = 0.1
                    self.re_prob = 0.5
                    self.margin = 0.0
                    self.lr_sched = "cosine"
                    self.lr_delay_epochs = 25
                    self.lr_eta_min = 7e-08
                elif "arbase" in self.backbone:
                    self.re_prob = 0.0          # REA disabled for ARBase
                    self.margin = 0.3
                    self.lr_sched = "cosine"
                    self.lr_delay_epochs = 25   # Cosine decay begins at epoch 25 of 50
                    self.lr_eta_min = 7e-08
                elif "agw" in self.backbone:
                    self.re_prob = 0.5
                    self.margin = 0.0           # Weighted soft-margin triplet
                    self.lr_sched = "multistep"
                    self.lr_milestones = (25, 35)
                    self.lr_gamma = 0.1
                else:  # BoT / default dinov2
                    self.re_prob = 0.5
                    self.margin = 0.3
                    self.lr_sched = "multistep"
                    self.lr_milestones = (25, 35)
                    self.lr_gamma = 0.1
            elif "mgn" in self.backbone:
                self.img_size = (384, 384)
                self.re_prob = 0.0          # REA disabled for MGN in OpenAnimals
                self.margin = 0.3           # OpenAnimals Triplet margin = 0.3
                self.epochs = 35
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.lr = 1.2e-04
                self.backbone_lr_factor = 1.0
                self.optimizer = "adam"
                self.weight_decay = 0.0005
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 25)
                self.lr_gamma = 0.1
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            elif "arbase" in self.backbone:
                self.img_size = (384, 384)
                self.re_prob = 0.0          # REA disabled for ARBase
                self.margin = 0.3
                self.epochs = 35
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.lr = 1.2e-04
                self.backbone_lr_factor = 1.0
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 25)
                self.lr_gamma = 0.1
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            elif "agw" in self.backbone:
                self.img_size = (256, 128)
                self.re_prob = 0.5          # REA enabled for AGW
                self.margin = 0.0           # Weighted soft-margin triplet
                self.epochs = 35
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.lr = 1.2e-04
                self.backbone_lr_factor = 1.0
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 25)
                self.lr_gamma = 0.1
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            elif "sbs" in self.backbone:
                self.img_size = (384, 128)
                self.re_prob = 0.5          # REA enabled for SBS
                self.autoaug_prob = 0.1     # AutoAugment p=0.1
                self.margin = 0.0           # Soft-margin triplet
                self.epochs = 35
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.freeze_epochs = 5
                self.lr = 1.2e-04
                self.backbone_lr_factor = 1.0
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 25)
                self.lr_gamma = 0.1
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            elif "bot" in self.backbone:
                self.img_size = (256, 128)
                self.re_prob = 0.5
                self.margin = 0.3
                self.epochs = 35
                self.warmup_epochs = 5
                self.warmup_factor = 0.1
                self.lr = 1.2e-04
                self.backbone_lr_factor = 1.0
                self.lr_sched = "multistep"
                self.lr_milestones = (15, 25)
                self.lr_gamma = 0.1
                self.chunk_size = 64
                self.backbone_drop_epoch = None
            else:
                self.img_size = (256, 128)
                self.re_prob = 0.5
                self.chunk_size = 64

            if getattr(self, "reid_method", None) is None:
                self.reid_method = "bot"
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
            self.optimizer = "adamw"
            self.weight_decay = 1e-05
            self.backbone_lr_factor = 1.0 if getattr(self, "full_finetune", False) else 0.1
            self.lr = 2e-05
            self.margin = 0.3
            self.epochs = 50
            self.warmup_epochs = 5
            self.warmup_factor = 0.01
            self.lr_sched = "multistep"
            self.lr_milestones = (25, 35)
            self.lr_gamma = 0.1
        elif "tfclip" in self.backbone or "tf_clip" in self.backbone or self.backbone == "clip":
            self.embedding_dim = 2048
            self.img_size = (224, 224)
            self.re_prob = 0.5
            self.batch_size = 16
            self.k = 4
            self.num_ids = 4
            self.chunk_size = 32
            self.optimizer = "adam"
            self.weight_decay = 1e-04
            self.lr = 5e-05
            self.backbone_lr_factor = 0.1
            self.margin = 0.3
            self.epochs = 80
            self.warmup_epochs = 5
            self.warmup_factor = 0.1
            self.lr_sched = "multistep"
            self.lr_milestones = (30, 50, 70)
            self.lr_gamma = 0.1
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
        clean_oa = bb.replace("openanimals_", "").replace("oa_", "")
        if clean_oa in ("bot", "arbase", "agw", "sbs", "mgn", "arbase_mb", "arbase_mgn"):
            base = clean_oa
            if backbone_lr_factor is not None and abs(float(backbone_lr_factor) - 1.0) > 1e-4:
                tune = f"{float(backbone_lr_factor):g}x"
                name = f"{base}_{tune}"
            else:
                name = base
        elif "dinov2" in bb:
            for v in ("dinov2_bot", "dinov2_agw", "dinov2_sbs", "dinov2_arbase"):
                if v in bb:
                    base = v
                    break
            else:
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
        bb = str(self.backbone).lower() if self.backbone else ""
        clean_oa = bb.replace("openanimals_", "").replace("oa_", "")
        is_resnet_model = ("resnet" in bb) or (clean_oa in ("bot", "arbase", "agw", "sbs", "mgn", "arbase_mb", "arbase_mgn") and "dinov2" not in bb)
        if "dinov2" in bb:
            ckpt_dir_name = "checkpoints_dinov2"
        elif is_resnet_model:
            ckpt_dir_name = "checkpoints_resnet"
        else:
            ckpt_dir_name = "checkpoints"
        model_folder = self.run_name

        # Multi-run subfolder support:
        # Hierarchy: checkpoints_<family>/<model_name>/<params>/
        run_name_override = getattr(self, "custom_run_name", None)
        if run_name_override:
            self.run_subfolder = str(run_name_override)
            self.output_dir = self.project_root / ckpt_dir_name / model_folder / self.run_subfolder
        else:
            if "dinov2" in bb:
                default_sub = "v1_cls_lr2e-5"
            elif is_resnet_model:
                ms_str = "-".join(str(m) for m in getattr(self, "lr_milestones", (15, 25)))
                default_sub = f"lr{self.lr:g}_ep{self.epochs}_ms{ms_str}_bb{bb_factor:g}x"
            else:
                default_sub = "default"
            self.run_subfolder = default_sub
            self.output_dir = self.project_root / ckpt_dir_name / model_folder / self.run_subfolder

        if make_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.run_name

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, list, tuple)):
                d[k] = list(v) if isinstance(v, tuple) else v
            elif isinstance(v, Path):
                d[k] = str(v)
            elif hasattr(v, "type"):
                d[k] = str(v)
        return d

    def save_config(self, path=None):
        import json
        if path is None:
            path = self.output_dir / "config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"[config] Saved run parameters to {path}")

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
