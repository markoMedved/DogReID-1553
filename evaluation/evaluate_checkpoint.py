import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from data.dataloader import build_dataloaders
from models.openanimals_models import build_oa_model


def calculate_cmc_map(distmat, q_pids, g_pids):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP = [], []
    for i in range(num_q):
        row_matches = matches[i]
        if not np.any(row_matches):
            continue
        cmc = np.cumsum(row_matches)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)

        d_idx = np.where(row_matches == 1)[0]
        num_rel = len(d_idx)
        precision_at_k = np.arange(1, num_rel + 1) / (d_idx + 1)
        all_AP.append(np.sum(precision_at_k) / num_rel)

    cmc_curve = np.mean(all_cmc, axis=0) if all_cmc else np.zeros(num_g)
    mAP = np.mean(all_AP) if all_AP else 0.0
    return cmc_curve[0], cmc_curve[4] if len(cmc_curve) > 4 else 0.0, mAP


def get_features(model, loader, device, desc="Extracting"):
    feats, pids = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            clips = batch[0].to(device)
            labels = batch[1]
            f = model(clips)
            if isinstance(f, tuple):
                f = f[0]
            f = F.normalize(f, p=2, dim=1)
            feats.append(f.cpu())
            pids.extend(labels.tolist())
    return torch.cat(feats, 0), np.array(pids)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DogReID checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--backbone", type=str, default="oa_bot")
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--pooling_type", type=str, default="mean")
    args = parser.parse_args()

    cfg = Config()
    cfg.backbone = args.backbone
    cfg.clip_len = args.clip_len
    cfg.batch_size = args.batch_size
    cfg.val_split = args.val_split
    cfg.pooling_type = args.pooling_type
    cfg.num_workers = 4
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {cfg.device}")

    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    num_classes = len(train_loader.dataset.dataset.id_map) if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'id_map') else 595
    from models.model_factory import build_model
    cfg.num_classes = num_classes
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.to(cfg.device)
    model.eval()

    q_f, q_pids = get_features(model, query_loader, cfg.device, "Query")
    g_f, g_pids = get_features(model, gallery_loader, cfg.device, "Gallery")

    distmat = 1.0 - torch.mm(q_f, g_f.t()).numpy()
    r1, r5, mAP = calculate_cmc_map(distmat, q_pids, g_pids)
    print("=" * 60)
    print(f"CHECKPOINT: {args.checkpoint}")
    print(f"CLEAN EVAL -> Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, mAP: {mAP:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
