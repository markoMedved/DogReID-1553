import argparse
import torch
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners

from models.freezing import apply_freezing, trainable_report
from solver.warmup import build_scheduler


def main():
    # ------------------------------------------------
    # ARGUMENT PARSING
    # ------------------------------------------------
    parser = argparse.ArgumentParser(description="Dog Re-ID Training")

    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--margin', type=float, default=None, help='Triplet loss margin')
    parser.add_argument('--weight_decay', type=float, default=None, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (P*K)')
    parser.add_argument('--k', type=int, default=None, help='Clips per dog ID')
    parser.add_argument('--model', type=str, default=None, help="Model/backbone: 'dinov2', 'bot', 'arbase', 'agw', 'sbs', 'mgn', 'tfclip'")
    parser.add_argument('--world', type=str, default=None, help="'closed' or 'open'")
    parser.add_argument('--clip_len', type=int, default=None, help='Frames per video clip')
    parser.add_argument('--use_images', action='store_true', default=False, help='Train on static images instead of video clips (Image-to-Image training)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--val_split', type=float, default=None, help='Validation split ratio (e.g., 0.2 for 20%% validation)')
    parser.add_argument('--eval_period', type=int, default=None, help='Evaluation frequency in epochs (default from config: 1)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers')

    # Re-ID methodology flags
    parser.add_argument(
        '--reid_method', 
        type=str, 
        default=None,
        choices=['bot', 'baseline'],
        help="Re-ID method: 'bot'"
    )
    parser.add_argument(
        '--pooling_type', 
        type=str, 
        default=None, 
        choices=['attention', 'mean', 'max'],
        help="Temporal aggregation method: 'attention', 'mean', or 'max'"
    )
    parser.add_argument(
        '--full_finetune', 
        dest='full_finetune', 
        action='store_true', 
        default=None, 
        help='Unfreeze the entire backbone for end-to-end fine-tuning'
    )
    parser.add_argument(
        '--no_full_finetune', 
        dest='full_finetune', 
        action='store_false', 
        help='Train only the last unfreeze_blocks backbone blocks'
    )
    parser.add_argument(
        '--unfreeze_blocks', 
        type=int, 
        default=None,
        help='Trailing backbone blocks to train when not full fine-tuning'
    )
    parser.add_argument(
        '--representation_type',
        type=str,
        default=None,
        choices=['cls', 'patch_mean', 'patch_gem', 'gem', 'cls_mean', 'cls_gem'],
        help="Feature representation for DINOv2: 'cls', 'patch_mean', 'patch_gem', 'cls_mean', or 'cls_gem'"
    )
    parser.add_argument(
        '--use_id_loss', 
        dest='use_id_loss', 
        action='store_true', 
        default=None, 
        help='Enable Identity Classification Loss alongside Triplet Loss'
    )
    parser.add_argument(
        '--no_use_id_loss', 
        dest='use_id_loss', 
        action='store_false', 
        help='Disable Identity Classification Loss'
    )
    parser.add_argument(
        '--use_bnneck',
        dest='use_bnneck',
        action='store_true',
        default=None,
        help='Enable BN-Neck in ReID model head (default: True)'
    )
    parser.add_argument(
        '--no_bnneck',
        dest='use_bnneck',
        action='store_false',
        help='Disable BN-Neck in ReID model head'
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=None,
        help='Label smoothing epsilon for ID loss (default: 0.1)'
    )
    parser.add_argument(
        '--no_label_smoothing', '--no_ls',
        dest='label_smoothing',
        action='store_const',
        const=0.0,
        help='Disable label smoothing for ID loss (sets label_smoothing=0.0)'
    )
    parser.add_argument(
        '--re_prob',
        type=float,
        default=None,
        help='Random Erasing probability (default: 0.5)'
    )
    parser.add_argument(
        '--no_re', '--no_random_erasing',
        dest='re_prob',
        action='store_const',
        const=0.0,
        help='Disable Random Erasing data augmentation (sets re_prob=0.0)'
    )
    parser.add_argument(
        '--autoaug_prob',
        type=float,
        default=None,
        help='AutoAugment probability (e.g. 0.1 for SBS AutoAug, default: 0.0)'
    )
    parser.add_argument(
        '--autoaug',
        dest='autoaug_prob',
        action='store_const',
        const=0.1,
        help='Enable SBS AutoAugment with probability 0.1'
    )
    parser.add_argument(
        '--mask_dog', 
        action='store_true', 
        default=False, 
        help='Mask out the dog for the background-only diagnostic experiment'
    )
    parser.add_argument(
        '--force_yolo',
        dest='force_yolo',
        action='store_true',
        default=True,
        help='Use YOLO dog detector dynamically to crop subjects (default: True)'
    )
    parser.add_argument(
        '--use_gt_boxes',
        dest='force_yolo',
        action='store_false',
        help='Use precomputed ground truth bounding boxes from CSV instead of YOLO detector'
    )
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default=None, 
        help='Optimizer to use: adam, adamw, or sgd'
    )
    parser.add_argument(
        '--accum_steps', 
        type=int, 
        default=None, 
        help='Number of gradient accumulation steps (default 1 for OpenAnimals)'
    )
    parser.add_argument(
        '--backbone_lr_factor',
        type=float,
        default=None,
        help='Multiplier for backbone LR relative to base LR (default: 1.0 to match OpenAnimals)'
    )
    parser.add_argument(
        '--backbone_drop_epoch',
        type=int,
        default=None,
        help='Epoch at which to drop backbone LR by 10x (default: model-dependent from config)'
    )
    parser.add_argument(
        '--backbone_drop_factor',
        type=float,
        default=None,
        help='Factor by which to drop backbone LR (default: 0.1)'
    )
    parser.add_argument(
        '--lr_sched',
        type=str,
        default=None,
        choices=['multistep', 'cosine'],
        help='Learning rate scheduler: multistep or cosine'
    )
    parser.add_argument(
        '--lr_delay_epochs',
        type=int,
        default=None,
        help='Delay epochs before cosine annealing begins'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Frame chunk size for video backbone forward pass'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=None,
        help='Number of linear warmup epochs'
    )
    parser.add_argument(
        '--warmup_factor',
        type=float,
        default=None,
        help='Initial LR multiplier at start of warmup'
    )
    parser.add_argument(
        '--lr_milestones',
        type=str,
        default=None,
        help='Comma-separated epoch milestones for MultiStepLR (e.g. "25,35")'
    )
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=True, 
        help='Resume training from the latest checkpoint if present (default: True)'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true', 
        default=False, 
        help='Explicitly wipe the checkpoint directory and restart from epoch 0'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Run name for organizing multiple experiment configurations under the model checkpoint folder (e.g. v1_cls_lr2e-5)'
    )
    parser.add_argument(
        '--fuse_layers',
        type=int,
        default=None,
        help='Number of trailing DINOv2 layers to fuse with learnable weights (default: 1)'
    )
    parser.add_argument(
        '--triplet_type',
        type=str,
        default=None,
        choices=['hard', 'soft', 'weighted'],
        help='Triplet loss variant: hard (hinge margin), soft (softplus batch-hard), or weighted (AGW WRT)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Explicit output directory for checkpoints and logs'
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()

    # --- Pre-update Model & Finetune Flags ---
    if args.model is not None:
        cfg.model = args.model
        cfg.backbone = args.model
    if args.full_finetune is not None:
        cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None:
        cfg.unfreeze_blocks = args.unfreeze_blocks
    if args.reid_method is not None:
        cfg.reid_method = None if args.reid_method == 'baseline' else args.reid_method
    if args.representation_type is not None:
        cfg.representation_type = args.representation_type

    cfg.update_model_settings()

    # --- Command-Line Overrides (applied after model defaults) ---
    if args.resume:
        cfg.resume = True
    if args.overwrite:
        cfg.overwrite = True
        cfg.resume = False
    if args.mask_dog:
        cfg.mask_dog = True
    if hasattr(args, "force_yolo") and args.force_yolo is not None:
        cfg.force_yolo = args.force_yolo
    if args.world is not None: cfg.world = args.world
    if args.clip_len is not None: cfg.clip_len = args.clip_len
    if args.use_images:
        cfg.use_videos = False
        cfg.clip_len = 1
    if args.lr is not None: cfg.lr = args.lr
    if args.margin is not None: cfg.margin = args.margin
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.k is not None: cfg.k = args.k
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.val_split is not None: cfg.val_split = args.val_split
    if args.eval_period is not None: cfg.eval_period = args.eval_period
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.accum_steps is not None: cfg.accum_steps = args.accum_steps
    if args.chunk_size is not None: cfg.chunk_size = args.chunk_size
    if args.warmup_epochs is not None: cfg.warmup_epochs = args.warmup_epochs
    if args.warmup_factor is not None: cfg.warmup_factor = args.warmup_factor
    if args.lr_milestones is not None:
        cfg.lr_milestones = tuple(int(x.strip()) for x in args.lr_milestones.split(",") if x.strip())
    if args.reid_method is not None:
        cfg.reid_method = None if args.reid_method == 'baseline' else args.reid_method
    if args.pooling_type is not None: cfg.pooling_type = args.pooling_type
    if args.full_finetune is not None: cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None: cfg.unfreeze_blocks = args.unfreeze_blocks
    if args.use_id_loss is not None: cfg.use_id_loss = args.use_id_loss
    if args.optimizer is not None: cfg.optimizer = args.optimizer
    if args.backbone_lr_factor is not None: cfg.backbone_lr_factor = args.backbone_lr_factor
    if args.lr_sched is not None: cfg.lr_sched = args.lr_sched
    if args.lr_delay_epochs is not None: cfg.lr_delay_epochs = args.lr_delay_epochs
    if args.backbone_drop_epoch is not None: cfg.backbone_drop_epoch = args.backbone_drop_epoch
    if args.backbone_drop_factor is not None: cfg.backbone_drop_factor = args.backbone_drop_factor
    if args.use_bnneck is not None: cfg.use_bnneck = args.use_bnneck
    if args.label_smoothing is not None: cfg.label_smoothing = args.label_smoothing
    if args.re_prob is not None: cfg.re_prob = args.re_prob
    if getattr(args, "autoaug_prob", None) is not None: cfg.autoaug_prob = args.autoaug_prob
    if getattr(args, "fuse_layers", None) is not None: cfg.fuse_layers = args.fuse_layers
    if getattr(args, "triplet_type", None) is not None: cfg.triplet_type = args.triplet_type
    if getattr(args, "run_name", None) is not None:
        cfg.custom_run_name = args.run_name
    cfg.refresh_run_name(make_dir=(getattr(args, "output_dir", None) is None))
    if hasattr(args, "output_dir") and args.output_dir is not None:
        from pathlib import Path
        cfg.output_dir = Path(args.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.display()
    cfg.save_config()

    # ------------------------------------------------
    # BUILD DATA LOADERS
    # ------------------------------------------------
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

    # Dynamic num_classes calculation
    if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'id_map'):
        cfg.num_classes = len(train_loader.dataset.dataset.id_map)
    elif hasattr(train_loader.dataset, 'dog_ids'):
        cfg.num_classes = len(set(train_loader.dataset.dog_ids))
    elif hasattr(train_loader.dataset, 'labels'):
        cfg.num_classes = len(set(train_loader.dataset.labels))
    elif hasattr(train_loader.dataset, 'pids'):
        cfg.num_classes = len(set(train_loader.dataset.pids))
    elif hasattr(train_loader.dataset, 'classes'):
        cfg.num_classes = len(train_loader.dataset.classes)
    else:
        cfg.num_classes = len(train_loader.dataset.targets if hasattr(train_loader.dataset, 'targets') else train_loader.dataset)

    print(f"--> Total training dog identities (num_classes): {cfg.num_classes}")

    # If id_loss is disabled for baseline, reset num_classes = 0 so the classifier head isn't built
    if not cfg.use_id_loss and cfg.reid_method != 'bot':
        cfg.num_classes = 0

    # ------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------
    model = build_model(cfg).to(cfg.device)

    # Apply freezing
    model = apply_freezing(model, cfg)
    print(trainable_report(model))

    # ------------------------------------------------
    # OPTIMIZER & SCHEDULER
    # ------------------------------------------------
    oa_sched_dict = None

    def is_pretrained(name):
        if 'NL_' in name or 'nonlocal' in name.lower() or 'gem' in name.lower() or 'layer_weight' in name.lower() or 'fusion' in name.lower():
            return False
        return (
            name.startswith('backbone')
            or name.startswith('base.')
            or name.startswith('jpm.')
            or name.startswith('image_encoder.')
            or name.startswith('oa_model.backbone')
            or name.startswith('oa_model.b1')
            or name.startswith('oa_model.b2')
            or name.startswith('oa_model.b3')
            or name.startswith('shared_base')
            or name.startswith('b1.')
            or name.startswith('b2.')
            or name.startswith('b3.')
        )

    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and is_pretrained(n)
    ]

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not is_pretrained(n)
    ]

    bb_lr_factor = float(getattr(cfg, "backbone_lr_factor", 1.0))
    if getattr(model, "is_tfclip", False) and hasattr(model, "get_parameter_groups"):
        param_groups = model.get_parameter_groups(
            base_lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            bias_lr_factor=2.0,
            large_fc_lr=True
        )
        n_bb = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("image_encoder."))
        n_hd = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("image_encoder."))
        print(f"[optim] TF-CLIP Stage-2 Groups | Backbone: {n_bb:,} params (LR={cfg.lr:.2e}) | TMD/SSP/Heads: {n_hd:,} params (Base LR={cfg.lr:.2e}, bias=2x, classifier_proj_temp=10x)")
    else:
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.lr * bb_lr_factor, "is_backbone": True})
        if head_params:
            param_groups.append({"params": head_params, "lr": cfg.lr, "is_backbone": False})

        if not param_groups:
            raise ValueError(
                "No trainable parameters found for optimizer! "
                "Ensure at least part of the backbone or head has requires_grad=True."
            )

        n_bb = sum(p.numel() for p in backbone_params)
        n_hd = sum(p.numel() for p in head_params)
        drop_info = f", drop 0.1x at epoch {cfg.backbone_drop_epoch}" if getattr(cfg, "backbone_drop_epoch", None) else ""
        print(f"[optim] Backbone: {n_bb:,} params, base LR = {cfg.lr * bb_lr_factor:.2e} ({bb_lr_factor}x{drop_info}) | Head: {n_hd:,} params, base LR = {cfg.lr:.2e}")
    opt_name = str(getattr(cfg, "optimizer", "adam")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=getattr(cfg, "weight_decay", 5e-4))
        print(f"[optim] Using SGD optimizer (momentum=0.9, weight_decay={getattr(cfg, 'weight_decay', 5e-4)})")
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        print(f"[optim] Using AdamW optimizer (weight_decay={cfg.weight_decay})")
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
        print(f"[optim] Using Adam optimizer (weight_decay={cfg.weight_decay})")

    # --- BUILD SCHEDULER ---
    scheduler = None
    oa_sched_dict = None
    if getattr(model, "is_openanimals", False):
        try:
            from openanimals.solver.build import build_lr_scheduler
            # Align warmup iters and schedule parameters to cfg
            iters_per_epoch = len(train_loader)
            warmup_epochs = getattr(cfg, "warmup_epochs", 5)
            model.oa_cfg.SOLVER.MAX_EPOCH = int(getattr(cfg, "epochs", model.oa_cfg.SOLVER.MAX_EPOCH))
            model.oa_cfg.SOLVER.BASE_LR = float(getattr(cfg, "lr", model.oa_cfg.SOLVER.BASE_LR))
            model.oa_cfg.SOLVER.WARMUP_ITERS = warmup_epochs * iters_per_epoch
            model.oa_cfg.SOLVER.WARMUP_FACTOR = getattr(cfg, "warmup_factor", 0.1)
            lr_sched_type = str(getattr(cfg, "lr_sched", "")).lower()
            if lr_sched_type == "multistep":
                model.oa_cfg.SOLVER.SCHED = "MultiStepLR"
                model.oa_cfg.SOLVER.STEPS = [int(m) for m in getattr(cfg, "lr_milestones", (15, 25))]
                model.oa_cfg.SOLVER.GAMMA = float(getattr(cfg, "lr_gamma", 0.1))
                model.oa_cfg.SOLVER.DELAY_EPOCHS = 0
            elif lr_sched_type == "cosine":
                model.oa_cfg.SOLVER.SCHED = "CosineAnnealingLR"
                model.oa_cfg.SOLVER.DELAY_EPOCHS = int(getattr(cfg, "lr_delay_epochs", model.oa_cfg.SOLVER.DELAY_EPOCHS))
                model.oa_cfg.SOLVER.ETA_MIN_LR = float(getattr(cfg, "lr_eta_min", model.oa_cfg.SOLVER.ETA_MIN_LR))
            oa_sched_dict = build_lr_scheduler(model.oa_cfg, optimizer, iters_per_epoch)
            sched_detail = (
                f"steps={model.oa_cfg.SOLVER.STEPS}"
                if model.oa_cfg.SOLVER.SCHED == "MultiStepLR"
                else f"delay={model.oa_cfg.SOLVER.DELAY_EPOCHS}"
            )
            print(f"[scheduler] Initialized OpenAnimals native scheduler: {model.oa_cfg.SOLVER.SCHED} ({sched_detail}, warmup: {warmup_epochs} epochs = {model.oa_cfg.SOLVER.WARMUP_ITERS} iters)")
        except Exception as e:
            print(f"[scheduler] Warning: failed to build OpenAnimals scheduler ({e}), falling back to WarmupMultiStepLR")
            scheduler = build_scheduler(optimizer, cfg)
    else:
        scheduler = build_scheduler(optimizer, cfg)

    # ------------------------------------------------
    # METRIC LEARNING SETUP
    # ------------------------------------------------
    miner = miners.BatchHardMiner()
    loss_fn = losses.TripletMarginLoss(margin=cfg.margin)

    # ------------------------------------------------
    # TRAINER OBJECT
    # ------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        miner=miner,
        cfg=cfg,
        scheduler=scheduler,
        oa_sched_dict=oa_sched_dict
    )

    trainer.train()


if __name__ == "__main__":
    main()