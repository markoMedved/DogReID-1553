import argparse
import os

import torch

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval


def main():
    parser = argparse.ArgumentParser(description="MetaN Evaluation")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument("--model_path", default="", type=str, help="path to trained weights")
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
    cfg.defrost()
    cfg.TEST.EVAL = True
    if args.model_path:
        cfg.TEST.WEIGHT = args.model_path
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("MetaN.Test", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Running with config:\n%s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if not cfg.TEST.WEIGHT:
        raise ValueError("Please provide --model_path or TEST.WEIGHT.")
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()

    for name, val_loader in val_loaders.items():
        evaluator = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        with torch.no_grad():
            for img, pids, cams, camids, viewids, img_paths in val_loader:
                img = img.to(device)
                feat, _ = model(forward_type="main", x=img)
                evaluator.update((feat, pids))

        cmc, mAP, mINP, *_ = evaluator.compute()
        logger.info("Validation Results - %s", name)
        logger.info("mAP: %.1f%%, mINP: %.1f%%", mAP * 100, mINP * 100)
        for rank in [1, 5, 10]:
            if len(cmc) >= rank:
                logger.info("CMC Rank-%d: %.1f%%", rank, cmc[rank - 1] * 100)


if __name__ == "__main__":
    main()
