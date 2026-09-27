#!/usr/bin/env python
# encoding: utf-8
"""
Evaluation script for DogReID-1553 using OpenAnimals models.
Computes both:
1. Native OpenAnimals CMC / mAP metrics
2. DogReID-1553 benchmark metrics (closed-set and open-set bootstrap evaluation via evaluation_utils.py)
3. Saves distance matrices to evaluation/csvs/ for direct compatibility with DogReID plotting notebooks.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add OpenAnimals and DogReID root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
OPENANIMALS_DIR = SCRIPT_DIR.parent
DOGREID_DIR = OPENANIMALS_DIR.parent
sys.path.insert(0, str(OPENANIMALS_DIR))
sys.path.insert(0, str(DOGREID_DIR))

from openanimals.config import get_cfg
from openanimals.engine import DefaultTrainer, default_argument_parser, default_setup
from openanimals.utils.checkpoint import Checkpointer
from openanimals.data import build_reid_test_loader
from evaluation.evaluation_utils import bootstrap_from_csv


def extract_features(model, dataloader, device):
    model.eval()
    all_feats = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["images"].to(device)
            feats = model(images)
            feats = F.normalize(feats, p=2, dim=1).cpu()
            all_feats.append(feats)
            all_paths.extend(batch["img_paths"])

    return torch.cat(all_feats, dim=0), all_paths


def path_to_id(img_path):
    # Extracts DOG_ID_VIDEO_ID from .../DOG_ID/DOG_ID-VIDEO_ID.jpg
    filename = Path(img_path).stem  # DOG_ID-VIDEO_ID
    parts = filename.split('-')
    dog_id = parts[0]
    video_id = '-'.join(parts[1:])
    return f"{dog_id}_{video_id}"


def evaluate_dogreid(cfg, model, split_type="closed", output_dir=".", model_tag="mgn", n_bootstrap=100):
    dataset_name = "DogReID" if split_type == "closed" else "DogReIDOpen"
    print(f"\n==========================================")
    print(f"Evaluating {dataset_name} ({split_type.upper()}-SET)")
    print(f"==========================================")

    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name)
    device = torch.device(cfg.MODEL.DEVICE)

    feats, paths = extract_features(model, test_loader, device)
    q_feats = feats[:num_query].to(device)
    g_feats = feats[num_query:].to(device)
    q_paths = paths[:num_query]
    g_paths = paths[num_query:]

    q_ids = [path_to_id(p) for p in q_paths]
    g_ids = [path_to_id(p) for p in g_paths]

    print(f"Computing distance matrix ({len(q_ids)} query x {len(g_ids)} gallery)...")
    dist_mat = torch.cdist(q_feats, g_feats, p=2).cpu().numpy()

    # Create DataFrame with format required by bootstrap_from_csv
    dist_df = pd.DataFrame(dist_mat, columns=g_ids)
    dist_df.insert(0, "queryId", q_ids)

    # 1. Save to model output dir
    os.makedirs(output_dir, exist_ok=True)
    local_csv_path = os.path.join(output_dir, f"dist_matrix_{split_type}.csv")
    dist_df.to_csv(local_csv_path, index=False)
    print(f"Distance matrix saved to: {local_csv_path}")

    # 2. Save directly into evaluation/csvs/ for notebook compatibility
    dogreid_csv_dir = DOGREID_DIR / "evaluation" / "csvs" / f"openanimals_{model_tag}_{split_type}_image"
    os.makedirs(dogreid_csv_dir, exist_ok=True)
    benchmark_csv_path = dogreid_csv_dir / f"{split_type}_dist_matrix.csv"
    dist_df.to_csv(benchmark_csv_path, index=False)
    print(f"Benchmark CSV saved to: {benchmark_csv_path}")

    # 3. Run DogReID bootstrap evaluation
    print(f"Running DogReID bootstrap evaluation ({n_bootstrap} iterations)...")
    res = bootstrap_from_csv(str(benchmark_csv_path), m=n_bootstrap, mode=split_type)

    print(f"\n--- DogReID Results ({split_type.upper()}) ---")
    if split_type == "closed":
        cmc = res.get("cmc_boot_mean")
        if cmc is None and "full_set" in res:
            cmc = res["full_set"].get("cmc")
        map_val = res.get("mAP_boot_mean")
        if map_val is None and "full_set" in res:
            map_val = res["full_set"].get("mAP")

        if cmc is not None:
            print(f"Rank-1:  {cmc[0]*100:.2f}%")
            print(f"Rank-5:  {cmc[4]*100:.2f}%")
            print(f"Rank-10: {cmc[9]*100:.2f}%")
        if map_val is not None:
            print(f"mAP:     {map_val*100:.2f}%")
    else:
        for k, v in res.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"{k}: {v['mean']:.4f} [95% CI: {v['ci_95'][0]:.4f} - {v['ci_95'][1]:.4f}]")

    return res


def main():
    parser = default_argument_parser()
    parser.add_argument("--eval-open", action="store_true", help="Also evaluate on open-set split")
    parser.add_argument("--bootstrap-iter", type=int, default=100, help="Number of bootstrap iterations")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print("Building model...")
    cfg_defrosted = cfg.clone()
    cfg_defrosted.defrost()
    cfg_defrosted.MODEL.BACKBONE.PRETRAIN = False
    model = DefaultTrainer.build_model(cfg_defrosted)

    weights_path = cfg.MODEL.WEIGHTS
    if not weights_path:
        # Prefer model_best.pth if it exists, otherwise fall back to model_final.pth
        best_candidate = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        final_candidate = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(best_candidate):
            weights_path = best_candidate
        elif os.path.exists(final_candidate):
            weights_path = final_candidate
        else:
            raise ValueError(f"Please specify MODEL.WEIGHTS or place model checkpoint in {cfg.OUTPUT_DIR}")

    print(f"Loading checkpoint from: {weights_path}")
    Checkpointer(model).load(weights_path)

    # Derive clean model tag (e.g. mgn, bot, agw, sbs, arbase)
    model_tag = Path(args.config_file).stem

    output_dir = os.path.join(cfg.OUTPUT_DIR, "dogreid_eval")

    # 1. Native FastReID / OpenAnimals test
    print("\n--- Running Native OpenAnimals Test ---")
    native_results = DefaultTrainer.test(cfg, model)

    # 2. DogReID Closed-Set Benchmark Test
    closed_res = evaluate_dogreid(
        cfg, model, split_type="closed", output_dir=output_dir, model_tag=model_tag, n_bootstrap=args.bootstrap_iter
    )

    # 3. DogReID Open-Set Benchmark Test (optional)
    if args.eval_open:
        open_res = evaluate_dogreid(
            cfg, model, split_type="open", output_dir=output_dir, model_tag=model_tag, n_bootstrap=args.bootstrap_iter
        )


if __name__ == "__main__":
    main()
