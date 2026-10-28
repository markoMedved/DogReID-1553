import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_train
from solver import create_scheduler, make_optimizer
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def is_main_process(local_rank):
    return (not dist.is_available()) or (not dist.is_initialized()) or local_rank == 0


def main():
    parser = argparse.ArgumentParser(description="MetaN Training")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using KEY VALUE pairs",
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0

    if is_main_process(local_rank):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    logger = setup_logger(
        "MetaN.Training",
        cfg.OUTPUT_DIR if is_main_process(local_rank) else None,
        if_train=True,
    )
    if is_main_process(local_rank):
        logger.info("Loaded config: %s", args.config_file)
        logger.info("Saving output to: %s", cfg.OUTPUT_DIR)
        logger.info("Running with config:\n%s", cfg)

    (
        train_loader,
        val_loaders,
        num_queries,
        num_classes,
        camera_num,
        view_num,
        species_num,
        num_images,
    ) = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, species_num=species_num)
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    start_epoch = 1
    if cfg.RESUME:
        checkpoint = torch.load(cfg.RESUME, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if is_main_process(local_rank):
            logger.info("Resumed from %s at epoch %d", cfg.RESUME, start_epoch)

    do_train(
        cfg=cfg,
        start_epoch=start_epoch,
        model=model,
        train_loader=train_loader,
        val_loaders=val_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_domains=species_num,
        num_classes=num_classes,
        num_images=num_images,
        num_queries=num_queries,
        local_rank=local_rank,
    )


if __name__ == "__main__":
    main()
