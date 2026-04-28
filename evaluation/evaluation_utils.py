import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

def _to_scalar(x):

    return x.item() if hasattr(x, "item") else x


def extract_features_with_ids(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:

    was_training = model.training
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []

    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                videos, _labels, dog_ids, video_ids = batch

                item_ids = [
                    f"{_to_scalar(d)}_{_to_scalar(v)}"
                    for d, v in zip(dog_ids, video_ids)
                ]

                # Normalize on device, then move to CPU immediately to free VRAM
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
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)

    print("-> Computing Distance Matrix...")
    dist_mat = torch.cdist(
        q_feat.to(cfg.device),
        g_feat.to(cfg.device),
        p=2,
    ).cpu().numpy()

    df = pd.DataFrame(dist_mat, columns=g_ids)
    df.insert(0, "queryId", q_ids)

    output_path = Path(cfg.output_dir) / filename
    df.to_csv(output_path, index=False)
    print(f"-> Distance CSV saved: {output_path}")

    return output_path

def bootstrap_from_csv(
    csv_path: str,
    m: int = 100,
    mode: str = "closed",
    random_state: int = 42,
) -> dict:
    
    if mode not in ("closed", "open"):
        raise ValueError(f"mode must be 'closed' or 'open', got '{mode}'")

    # --- Load ---
    df_full = pd.read_csv(csv_path)

    # dogId_videoId format → split from the right to handle underscores in IDs
    get_id = lambda x: str(x).rsplit('_', 1)[0]

    query_labels   = np.array([get_id(i) for i in df_full["queryId"].values])
    gallery_labels = np.array([get_id(i) for i in df_full.columns[1:].values])
    dist_mat       = df_full.iloc[:, 1:].values.astype(np.float32)

    # --- Identity index map for fast bootstrap sampling ---
    unique_ids     = np.unique(query_labels)
    id_to_indices  = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}

    # --- Precompute sorted match matrix (closed only) ---
    if mode == "closed":
        print("-> Pre-calculating sorted match matrix...")
        match_matrix   = query_labels[:, None] == gallery_labels[None, :]
        sort_idx       = np.argsort(dist_mat, axis=1)
        sorted_matches = np.take_along_axis(match_matrix, sort_idx, axis=1)

    rng = np.random.default_rng(random_state)
    boot_results = []

    print(f"-> Bootstrapping '{mode}' metrics from CSV ({m} iterations)...")

    for _ in tqdm(range(m)):
        sampled_ids      = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        selected_indices = np.concatenate([id_to_indices[id_] for id_ in sampled_ids])

        if mode == "closed":
            res = _calc_closed_logic(sorted_matches[selected_indices])
        else:
            res = _calc_open_logic(
                dist_mat[selected_indices],
                query_labels[selected_indices],
                gallery_labels,
            )
        boot_results.append(res)

    return _aggregate_bootstrap_results(boot_results, mode)


def _calc_closed_logic(matches: np.ndarray) -> dict:
    num_gallery = matches.shape[1]

    # Filter out queries that have no correct match in the gallery
    has_match    = np.any(matches, axis=1)
    valid_matches = matches[has_match]
    n_valid      = len(valid_matches)

    if n_valid == 0:
        return {"mAP": 0.0, "cmc": np.zeros(num_gallery)}

    # --- CMC ---
    # Rank of first correct match for each valid query (0-indexed)
    first_match_ranks = np.argmax(valid_matches, axis=1)
    cmc_counts        = np.bincount(first_match_ranks, minlength=num_gallery)
    cmc               = np.cumsum(cmc_counts) / n_valid

    # --- mAP ---
    # Precision at each rank, averaged over positions where a match occurs
    cum_matches = np.cumsum(valid_matches, axis=1)
    precisions  = cum_matches / np.arange(1, num_gallery + 1)
    n_relevant  = np.sum(valid_matches, axis=1)                          # per-query match count
    ap          = np.sum(precisions * valid_matches, axis=1) / n_relevant  # safe: n_valid > 0 and has_match guarantees n_relevant >= 1

    return {
        "mAP": float(np.mean(ap)),
        "cmc": cmc,
    }


# ---------------------------------------------------------------------------

_DIR_FAR_TARGETS  = (0.01, 0.05, 0.1)
_FAR_TOLERANCE    = 0.001
_N_THRESHOLDS     = 10_000


def _calc_open_logic(dist_mat: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray) -> dict:
    known_mask   = np.isin(q_labels, g_labels)
    unknown_mask = ~known_mask

    best_dist     = np.min(dist_mat, axis=1)        
    best_idx      = np.argmin(dist_mat, axis=1)      
    correct_match = g_labels[best_idx] == q_labels    

    thresholds = np.linspace(0, 2, _N_THRESHOLDS)  

    known_dists   = best_dist[known_mask]             
    known_correct = correct_match[known_mask]         
    unknown_dists = best_dist[unknown_mask]           

    n_known   = known_mask.sum()
    n_unknown = unknown_mask.sum()

    # --- Corrected Vectorized Logic ---
    if n_known > 0:
        # DIR: Correct identity AND distance below threshold
        under_and_correct = (known_dists[:, None] <= thresholds) & known_correct[:, None]
        dirs_at_t = under_and_correct.sum(axis=0) / n_known  
    else:
        dirs_at_t = np.zeros(_N_THRESHOLDS)

    if n_unknown > 0:
        # FAR: Any "unknown" dog that falls below the distance threshold (false alarm)
        under_unknown = unknown_dists[:, None] <= thresholds
        fars_at_t = under_unknown.sum(axis=0) / n_unknown  # FIXED: Removed the zero overwrite
    else:
        fars_at_t = np.zeros(_N_THRESHOLDS)

    res = {"dirs": dirs_at_t, "fars": fars_at_t}

    # Interpolate DIR at specific FAR points
    for target in _DIR_FAR_TARGETS:
        # Find the threshold index where FAR is closest to our target (e.g., 0.01)
        idx = np.argmin(np.abs(fars_at_t - target))
        achieved_far = fars_at_t[idx]
        
        # Check if we actually reached the target FAR within tolerance
        if np.abs(achieved_far - target) <= _FAR_TOLERANCE:
            res[target] = float(dirs_at_t[idx])
        else:
            res[target] = float("nan")

    return res


# ---------------------------------------------------------------------------

def _aggregate_bootstrap_results(results: list[dict], mode: str) -> dict:
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
    all_dirs = np.array([r["dirs"] for r in results])  
    all_fars = np.array([r["fars"] for r in results])  

    return {
        "mean_fars":  np.mean(all_fars,  axis=0),
        "mean_dirs":  np.mean(all_dirs,  axis=0),
        "lower_dirs": np.percentile(all_dirs, 2.5,  axis=0),
        "upper_dirs": np.percentile(all_dirs, 97.5, axis=0),
    }