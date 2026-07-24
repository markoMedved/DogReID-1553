import torch
import argparse
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners


def main():

    # ------------------------------------------------
    # ARGUMENT PARSING
    # Allows overriding config values from CLI
    # Example:
    # python train.py --model dinov2 --lr 1e-4 --batch_size 32
    # ------------------------------------------------
    parser = argparse.ArgumentParser(description="Dog Re-ID Training")

    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--margin', type=float, default=None, help='Triplet loss margin')
    parser.add_argument('--weight_decay', type=float, default=None, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (P*K)')
    parser.add_argument('--k', type=int, default=None, help='Clips per dog ID')
    parser.add_argument('--model', type=str, default=None, help="Backbone: 'dinov2', 'swin', 'vit', 'convnetxt'")
    parser.add_argument('--world', type=str, default=None, help="'closed' or 'open'")
    parser.add_argument('--clip_len', type=int, default=None, help='Frames per video clip')
    # Toggle full backbone fine-tuning via CLI flag
    parser.add_argument(
        '--pooling_type', 
        type=str, 
        default='attention', 
        choices=['attention', 'mean', 'max'],
        help="Temporal aggregation method: 'attention', 'mean', or 'max'"
    )
    parser.add_argument(
        '--full_finetune', 
        action='store_true', 
        default=False, 
        help='Unfreeze the entire backbone for end-to-end fine-tuning'
    )

    args = parser.parse_args()


    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()


    # ------------------------------------------------
    # OVERRIDE CONFIG WITH CLI ARGUMENTS
    # Only overwrite values that were provided
    # ------------------------------------------------
    if args.model: cfg.model = args.model
    if args.world: cfg.world = args.world
    if args.clip_len: cfg.clip_len = args.clip_len
    if args.lr: cfg.lr = args.lr
    if args.margin: cfg.margin = args.margin
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.k: cfg.k = args.k

    # Store full_finetune in config
    cfg.pooling_type = args.pooling_type
    cfg.full_finetune = args.full_finetune

    cfg.run_name = f"{cfg.model}_{cfg.world}_{cfg.pooling_type}_finetune_{cfg.full_finetune}"
    cfg.output_dir = cfg.project_root / "trained_models" / cfg.run_name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


    # print final configuration
    cfg.display()


    # ------------------------------------------------
    # BUILD DATA LOADERS
    # ------------------------------------------------
    # ------------------------------------------------
    # BUILD DATA LOADERS
    # ------------------------------------------------
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

    # Automatically count unique dog IDs in training set
    # (Adjust dataset attribute name if yours uses .num_identities or .classes)
    if hasattr(train_loader.dataset, 'pids'):
        cfg.num_classes = len(set(train_loader.dataset.pids))
    elif hasattr(train_loader.dataset, 'classes'):
        cfg.num_classes = len(train_loader.dataset.classes)
    else:
        # Standard fallback for ImageFolder style datasets
        cfg.num_classes = len(train_loader.dataset.targets if hasattr(train_loader.dataset, 'targets') else train_loader.dataset)
        
    print(f"--> Total training dog identities (num_classes): {cfg.num_classes}")


    # ------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------
    model = build_model(cfg).to(cfg.device)

    if cfg.full_finetune:
        print("--> Unfreezing FULL backbone for end-to-end fine-tuning.")
        for p in model.parameters():
            p.requires_grad = True
    else:
        # ------------------------------------------------
        # FREEZE ENTIRE MODEL FIRST
        # We selectively unfreeze layers afterwards
        # ------------------------------------------------
        for p in model.parameters():
            p.requires_grad = False


        # ------------------------------------------------
        # ARCHITECTURE-AWARE PARTIAL UNFREEZING
        # Each backbone has a slightly different internal structure
        # ------------------------------------------------
        # --- ConvNeXt Architecture ---
        # ConvNeXt feature extraction stages are stored in backbone.stages
        if hasattr(model.backbone, 'stages'):
            print("--> Unfreezing final ConvNeXt stage.")
            for p in model.backbone.stages[-1].parameters():
                p.requires_grad = True

            # Unfreeze layer norm / pre-norm if present
            if hasattr(model.backbone, 'norm_pre'):
                for p in model.backbone.norm_pre.parameters():
                    p.requires_grad = True

        # --- Torchvision ViT ---
        # encoder.layers = transformer blocks
        elif hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layers'):

            # unfreeze last 2 transformer blocks
            for layer in model.backbone.encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True

            # unfreeze final layer normalization
            if hasattr(model.backbone.encoder, 'ln'):
                for p in model.backbone.encoder.ln.parameters():
                    p.requires_grad = True


        # --- DINOv2 ---
        # transformer blocks stored as backbone.blocks
        elif hasattr(model.backbone, 'blocks'):

            for block in model.backbone.blocks[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

            # final normalization layer
            if hasattr(model.backbone, 'norm'):
                for p in model.backbone.norm.parameters():
                    p.requires_grad = True


        # --- Swin Transformer ---
        # hierarchical transformer stages
        elif hasattr(model.backbone, 'layers'):

            # unfreeze final stage
            for p in model.backbone.layers[-1].parameters():
                p.requires_grad = True

            if hasattr(model.backbone, 'norm'):
                for p in model.backbone.norm.parameters():
                    p.requires_grad = True


    # ------------------------------------------------
    # ALWAYS TRAIN TEMPORAL HEAD
    # (Temporal attention pooling)
    # ------------------------------------------------
    pool_layer = getattr(model, 'temporal_pool', getattr(model, 'temporal_attn', None))

    if pool_layer is not None and hasattr(pool_layer, 'parameters'):
        for p in pool_layer.parameters():
            p.requires_grad = True


    # BN neck is also always trainable
    if hasattr(model, 'bn') and model.bn is not None:
        for p in model.bn.parameters():
            p.requires_grad = True

    # ------------------------------------------------
    # OPTIMIZER
    # Different learning rates for backbone vs head
    # ------------------------------------------------
    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'backbone' not in n
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr * 0.1})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr})

    # Guard against passing an empty parameter list to AdamW
    if not param_groups:
        raise ValueError(
            "No trainable parameters found for optimizer! "
            "Ensure at least part of the backbone or head has requires_grad=True."
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    # ------------------------------------------------
    # METRIC LEARNING SETUP
    # ------------------------------------------------

    # hard triplet mining within batch
    miner = miners.BatchHardMiner()

    # triplet margin loss
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
        cfg=cfg
    )


    # ------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()