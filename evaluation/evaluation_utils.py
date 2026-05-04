import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

def _to_scalar(x):
    """Get scalar"""
    return x.item() if hasattr(x, "item") else x


def extract_features_with_ids(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Extract features of the samples from the provided ids"""

    # If the model was training, to restore back after
    was_training = model.training
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []

    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")

    try:
        with torch.no_grad():
            # --- Test loop ---
            for batch in tqdm(dataloader):
                videos, _labels, dog_ids, video_ids = batch

                item_ids = [
                    f"{_to_scalar(d)}_{_to_scalar(v)}"
                    for d, v in zip(dog_ids, video_ids)
                ]

                # Normalize on device, then move to CPU the features that the model produced
                embeddings = F.normalize(model(videos.to(device)), p=2, dim=1).cpu()

                all_embeddings.append(embeddings)
                all_ids.extend(item_ids)
    finally:
        model.train(was_training)  # always restore original mode

    return torch.cat(all_embeddings), all_ids


def generate_distance_csv(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    cfg,
    filename: str = "dist_matrix.csv",
) -> Path:
    """Generate a distance csv from features gotten with model inference"""

    # --- Get teh features for gallery and query samples ---
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)

    # --- Compute the distance matrix ---
    print("-> Computing Distance Matrix...")
    dist_mat = torch.cdist(
        q_feat.to(cfg.device),
        g_feat.to(cfg.device),
        p=2,
    ).cpu().numpy()

    # --- Construct the appropriate csv --- 
    df = pd.DataFrame(dist_mat, columns=g_ids)
    df.insert(0, "queryId", q_ids)

    # --- Create the output path ---
    output_path = Path(cfg.output_dir) / filename
    df.to_csv(output_path, index=False)
    print(f"-> Distance CSV saved: {output_path}")

    return output_path

def bootstrap_from_csv(
    csv_path: str,
    m: int = 100,
    mode: str = "closed",
    random_state: int = 42,
    # # scale: float = 1.0, 
    # # shift: float = 0.0,  
) -> dict:
    """Use boostrap for uncertenty quantification"""
    
    if mode not in ("closed", "open"):
        raise ValueError(f"mode must be 'closed' or 'open', got '{mode}'")

    # --- Load ---
    df_full = pd.read_csv(csv_path)

    # dogId_videoId format → split from the right to handle underscores in IDs
    get_id = lambda x: str(x).rsplit('_', 1)[0]

    query_labels   = np.array([get_id(i) for i in df_full["queryId"].values])
    gallery_labels = np.array([get_id(i) for i in df_full.columns[1:].values])

    # Scaling, just to test
    dist_mat       = df_full.iloc[:, 1:].values.astype(np.float32)
    # dist_mat       = (dist_mat * scale) + shift

    # --- Identity index map for fast bootstrap sampling ---
    unique_ids     = np.unique(query_labels)
    id_to_indices  = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}

    # --- Precompute sorted match matrix (closed only) ---
    if mode == "closed":
        print("-> Pre-calculating sorted match matrix...")
        match_matrix   = query_labels[:, None] == gallery_labels[None, :]
        sort_idx       = np.argsort(dist_mat, axis=1)
        sorted_matches = np.take_along_axis(match_matrix, sort_idx, axis=1)

    if mode == "open":
        global_thresholds = np.linspace(dist_mat.min(), dist_mat.max(), _N_THRESHOLDS)
    else:
        global_thresholds = None

    rng = np.random.default_rng(random_state)
    boot_results = []

    print(f"-> Bootstrapping '{mode}' metrics from CSV ({m} iterations)...")

    # --- Do m bootstrap repetitions ---
    for _ in tqdm(range(m)):
        # Sample by Dog ID not by videos
        sampled_ids      = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        selected_indices = np.concatenate([id_to_indices[id_] for id_ in sampled_ids])

        # Based on the setting calculate the appropriate metrics
        if mode == "closed":
            res = _calc_closed_logic(sorted_matches[selected_indices])
        else:
            res = _calc_open_logic(
                dist_mat[selected_indices],
                query_labels[selected_indices],
                gallery_labels,
                global_thresholds
            )
        boot_results.append(res)

    return _aggregate_bootstrap_results(boot_results, mode)


def _calc_closed_logic(matches: np.ndarray) -> dict:
    """Calculate cmc and mAP"""

    num_gallery = matches.shape[1]
    num_query = matches.shape[0]

    # --- CMC ---
    # Rank of first correct match for each valid query (0-indexed)
    first_match_ranks = np.argmax(matches, axis=1)
    cmc_counts        = np.bincount(first_match_ranks, minlength=num_gallery)
    cmc               = np.cumsum(cmc_counts) / num_query

    # --- mAP ---
    # Precision at each rank, averaged over positions where a match occurs
    cum_matches = np.cumsum(matches, axis=1)
    precisions  = cum_matches / np.arange(1, num_gallery + 1)
    n_relevant  = np.sum(matches, axis=1)                          
    ap          = np.sum(precisions * matches, axis=1) / n_relevant  

    return {
        "mAP": float(np.mean(ap)),
        "cmc": cmc,
    }


# ---------------------------------------------------------------------------

# Hyperparameters for evaluating in the open-world setting 
_DIR_FAR_TARGETS  = (0.01, 0.05, 0.1)
_FAR_TOLERANCE    = 0.001
_N_THRESHOLDS     = 1000


def _calc_open_logic(dist_mat: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray, thresholds: np.ndarray) -> dict:
    """Calculate DIR vs FAR for Rank1 and Rank5"""
    known_mask   = np.isin(q_labels, g_labels)
    unknown_mask = ~known_mask

    # Get Top-5 indices 
    top5_idx = np.argsort(dist_mat, axis=1)[:, :5]
    
    # Minimum distance (Rank-1)
    best_dist = dist_mat[np.arange(len(dist_mat)), top5_idx[:, 0]]
    
    # Rank-1 Match Logic
    best_idx = top5_idx[:, 0]
    correct_match_r1 = g_labels[best_idx] == q_labels    

    # Rank-5 Match Logic: Check if the correct label is in the top 5
    top5_labels = g_labels[top5_idx]
    correct_match_r5 = np.any(top5_labels == q_labels[:, None], axis=1)

    # thresholds = np.linspace(dist_mat.min(), dist_mat.max(), _N_THRESHOLDS) 
    # print(dist_mat.min(), dist_mat.max())

    # Separate Knowns
    known_dists      = best_dist[known_mask]             
    known_correct_r1 = correct_match_r1[known_mask]        
    known_correct_r5 = correct_match_r5[known_mask]

    # Separate Unknowns
    unknown_dists = best_dist[unknown_mask]           

    n_known   = known_mask.sum()
    n_unknown = unknown_mask.sum()

    if n_known > 0:
        # DIR @ Rank-1
        under_and_correct_r1 = (known_dists[:, None] <= thresholds) & known_correct_r1[:, None]
        dirs_at_t_r1 = under_and_correct_r1.sum(axis=0) / n_known  
        
        # DIR @ Rank-5
        under_and_correct_r5 = (known_dists[:, None] <= thresholds) & known_correct_r5[:, None]
        dirs_at_t_r5 = under_and_correct_r5.sum(axis=0) / n_known  
    else:
        dirs_at_t_r1 = np.zeros(_N_THRESHOLDS)
        dirs_at_t_r5 = np.zeros(_N_THRESHOLDS)

    if n_unknown > 0:
        # FAR: Any "unknown" dog that falls below the distance threshold (false alarm)
        under_unknown = unknown_dists[:, None] <= thresholds
        fars_at_t = under_unknown.sum(axis=0) / n_unknown  
    else:
        fars_at_t = np.zeros(_N_THRESHOLDS)

    res = {
        "dirs_r1": dirs_at_t_r1, 
        "dirs_r5": dirs_at_t_r5, 
        "fars": fars_at_t
    }

    # Interpolate DIR at specific FAR points
    for target in _DIR_FAR_TARGETS:
        idx = np.argmin(np.abs(fars_at_t - target))
        achieved_far = fars_at_t[idx]
        
        if np.abs(achieved_far - target) <= _FAR_TOLERANCE:
            # Save both R1 and R5 
            res[f"r1_{target}"] = float(dirs_at_t_r1[idx])
            res[f"r5_{target}"] = float(dirs_at_t_r5[idx])
        else:
            res[f"r1_{target}"] = float("nan")
            res[f"r5_{target}"] = float("nan")

    return res


# ---------------------------------------------------------------------------


def _aggregate_bootstrap_results(results: list[dict], mode: str) -> dict:
    """Return the results for closed and open world evaluation"""
    if mode == "closed":
        maps = np.array([r["mAP"] for r in results])
        cmcs = np.array([r["cmc"] for r in results])  

        mean_cmc = np.mean(cmcs, axis=0)

        return {
            "mAP_mean":  float(np.mean(maps)),
            "mAP_std":   float(np.std(maps)),
            "mAP_lower": float(np.percentile(maps, 2.5)),
            "mAP_upper": float(np.percentile(maps, 97.5)),
            "cmc_mean":  mean_cmc,
            "cmc_lower": np.percentile(cmcs, 2.5,  axis=0),
            "cmc_upper": np.percentile(cmcs, 97.5, axis=0),
            "ranks":     np.arange(1, len(mean_cmc) + 1),
        }

    # --- open ---
    all_dirs_r1 = np.array([r["dirs_r1"] for r in results])  
    all_dirs_r5 = np.array([r["dirs_r5"] for r in results])  
    all_fars    = np.array([r["fars"] for r in results])  

    aggregated_res = {
        "mean_fars":     np.mean(all_fars, axis=0),
        
        "mean_dirs_r1":  np.mean(all_dirs_r1, axis=0),
        "lower_dirs_r1": np.percentile(all_dirs_r1, 2.5, axis=0),
        "upper_dirs_r1": np.percentile(all_dirs_r1, 97.5, axis=0),
        
        "mean_dirs_r5":  np.mean(all_dirs_r5, axis=0),
        "lower_dirs_r5": np.percentile(all_dirs_r5, 2.5, axis=0),
        "upper_dirs_r5": np.percentile(all_dirs_r5, 97.5, axis=0),
    }

    # Automatically aggregate any targeted FAR points
    target_keys = [k for k in results[0].keys() if k.startswith("r1_") or k.startswith("r5_")]
    for tk in target_keys:
        vals = np.array([r[tk] for r in results])
        # Filter out NaNs in case some bootstrap samples didn't reach the target FAR
        valid_vals = vals[~np.isnan(vals)] 
        
        if len(valid_vals) > 0:
            aggregated_res[f"{tk}_mean"]  = float(np.mean(valid_vals))
            aggregated_res[f"{tk}_lower"] = float(np.percentile(valid_vals, 2.5))
            aggregated_res[f"{tk}_upper"] = float(np.percentile(valid_vals, 97.5))
        else:
            aggregated_res[f"{tk}_mean"]  = float("nan")
            aggregated_res[f"{tk}_lower"] = float("nan")
            aggregated_res[f"{tk}_upper"] = float("nan")

    return aggregated_res