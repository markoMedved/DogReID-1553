import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

def _to_scalar(x):
    """Convert a 0-d tensor or Python scalar to a plain Python value."""
    return x.item() if hasattr(x, "item") else x


def extract_features_with_ids(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """
    Run inference on all batches and return L2-normalised embeddings
    together with their 'dogId_videoId' string identifiers.

    Args:
        model:      Model whose forward() returns (N, D) embeddings.
        dataloader: Yields (videos, labels, dog_ids, video_ids) batches.
        device:     Device to run inference on.
    Returns:
        Tuple of (embeddings [N, D] float32 CPU tensor, list of N id strings).
    """
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
    """
    Extract embeddings for query and gallery sets, compute the L2 distance
    matrix, and save it as a CSV.

    Args:
        model:          Trained embedding model.
        query_loader:   DataLoader for query set.
        gallery_loader: DataLoader for gallery set.
        cfg:            Config object with .device and .output_dir attributes.
        filename:       Output CSV filename.
    Returns:
        Path to the saved CSV file.
    """
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
    """
    Bootstrap CMC/mAP (closed) or DIR@FAR (open) metrics from a precomputed
    distance matrix CSV.

    Args:
        csv_path:     Path to CSV with columns [queryId, galleryId_0, galleryId_1, ...]
        m:            Number of bootstrap iterations.
        mode:         'closed' or 'open'.
        random_state: Seed for reproducibility.
    Returns:
        dict of aggregated bootstrap statistics.
    """
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
    """
    Compute CMC and mAP for closed-world re-ID evaluation.

    Args:
        matches: Boolean array (NumQueries, NumGallery) sorted by ascending distance.
                 matches[i, j] = True if gallery item j is the correct match for query i.
    Returns:
        dict with keys: 'mAP' (float), 'cmc' (np.ndarray of length NumGallery)
    """
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
    """
    Compute DIR@FAR curve and DIR values at standard FAR operating points
    for open-world re-ID evaluation.

    Args:
        dist_mat: (NumQueries, NumGallery) pairwise distance matrix.
        q_labels: (NumQueries,) identity labels for queries.
        g_labels: (NumGallery,) identity labels for gallery items.
    Returns:
        dict with keys:
            'dirs'         – DIR curve, shape (N_THRESHOLDS,)
            'fars'         – FAR curve, shape (N_THRESHOLDS,)
            0.01, 0.05, 0.1 – DIR at each FAR target (float or NaN if unreachable)
    """
    known_mask   = np.isin(q_labels, g_labels)
    unknown_mask = ~known_mask

    best_dist    = np.min(dist_mat, axis=1)           # (NumQueries,)
    best_idx     = np.argmin(dist_mat, axis=1)        # (NumQueries,)
    correct_match = g_labels[best_idx] == q_labels    # (NumQueries,)

    thresholds = np.linspace(0, 2, _N_THRESHOLDS)    # (T,)

    # --- Vectorized DIR and FAR curves ---
    known_dists   = best_dist[known_mask]             # (K,)
    known_correct = correct_match[known_mask]         # (K,)
    unknown_dists = best_dist[unknown_mask]           # (U,)

    n_known   = known_mask.sum()
    n_unknown = unknown_mask.sum()

    if n_known > 0:
        # (K, T) broadcast: is each known query under threshold AND correct?
        under_and_correct = (known_dists[:, None] < thresholds) & known_correct[:, None]
        dirs_at_t = under_and_correct.sum(axis=0) / n_known   # (T,)
    else:
        dirs_at_t = np.zeros(_N_THRESHOLDS)

    if n_unknown > 0:
        # (U, T) broadcast: is each unknown query under threshold?
        under_unknown = unknown_dists[:, None] < thresholds
        fars_at_t = under_unknown.sum(axis=0) / n_unknown     # (T,)
    else:
        fars_at_t = np.zeros(_N_THRESHOLDS)

    # --- DIR at standard FAR operating points ---
    res = {"dirs": dirs_at_t, "fars": fars_at_t}

    for target in _DIR_FAR_TARGETS:
        idx      = np.argmin(np.abs(fars_at_t - target))
        achieved = fars_at_t[idx]
        res[target] = (
            float(dirs_at_t[idx])
            if np.abs(achieved - target) <= _FAR_TOLERANCE
            else float("nan")
        )

    return res


# ---------------------------------------------------------------------------

def _aggregate_bootstrap_results(results: list[dict], mode: str) -> dict:
    """
    Aggregate per-iteration bootstrap results into means and 95% CIs.

    Args:
        results: List of dicts returned by _calc_closed_logic or _calc_open_logic.
        mode:    'closed' or 'open'.
    Returns:
        dict of aggregated statistics.
    """
    if mode == "closed":
        maps = np.array([r["mAP"] for r in results])
        cmcs = np.array([r["cmc"] for r in results])   # (M, NumGallery)

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
    all_dirs = np.array([r["dirs"] for r in results])  # (M, T)
    all_fars = np.array([r["fars"] for r in results])  # (M, T)

    return {
        "mean_fars":  np.mean(all_fars,  axis=0),
        "mean_dirs":  np.mean(all_dirs,  axis=0),
        "lower_dirs": np.percentile(all_dirs, 2.5,  axis=0),
        "upper_dirs": np.percentile(all_dirs, 97.5, axis=0),
    }