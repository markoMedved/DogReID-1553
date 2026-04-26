import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

def extract_features_with_ids(model, dataloader, device):
    # --- Set Model to Evaluation Mode ---
    model.eval()
    all_embeddings = []
    all_ids = []

    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")

    # --- Feature Extraction Loop ---
    with torch.no_grad():
        for batch in tqdm(dataloader):
            videos, labels, dog_ids, video_ids = batch
            
            # Create unique item identifiers
            item_ids = [f"{d}_{v}" for d, v in zip(dog_ids, video_ids)]
            videos = videos.to(device)

            # Forward pass to get embeddings
            embeddings = model(videos)
            
            # L2 Normalize embeddings for cosine similarity equivalence
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())
            all_ids.extend(item_ids)

    return torch.cat(all_embeddings), all_ids

def generate_distance_csv(model, query_loader, gallery_loader, cfg, filename="dist_matrix.csv", sep=','):
    # --- Extract Features ---
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)

    print("-> Computing Distance Matrix...")
    dist_mat = torch.cdist(q_feat, g_feat, p=2).numpy()

    # --- Format as DataFrame ---
    df = pd.DataFrame(dist_mat, columns=g_ids)
    df.insert(0, 'queryId', q_ids)

    # --- Save to CSV ---
    output_path = Path(cfg.output_dir) / filename
    
    # Save with the specified separator (default is comma)
    df.to_csv(output_path, sep=sep, index=False)
    print(f"Distance CSV successfully created: {output_path} (Separator: '{sep}')")
    return output_path

def bootstrap_from_csv(csv_path, m=100, mode="closed", random_state=42, sep=None):
    """
    sep: if None, pandas will attempt to auto-detect the separator. 
         Otherwise, you can pass ',' or ';' explicitly.
    """
    # --- Load Distance Matrix ---
    # Using 'python' engine with sep=None allows pandas to auto-detect the delimiter
    if sep is None:
        df_full = pd.read_csv(csv_path, sep=None, engine='python')
    else:
        df_full = pd.read_csv(csv_path, sep=sep)
    
    # Rest of the function remains the same...
    get_dog_label = lambda x: str(x).split('_')[0]
    query_labels = np.array([get_dog_label(i) for i in df_full['queryId'].values])
    unique_ids = np.unique(query_labels)
    id_to_indices = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}

    # --- Pre-calculate Sorted Match Matrix (CRITICAL FOR SPEED) ---
    # This identifies where the "hits" are for every query, sorted by distance
    print("-> Pre-calculating sorted match matrix...")
    match_matrix = (query_labels[:, None] == gallery_labels[None, :])
    sort_idx = np.argsort(dist_mat, axis=1)
    # This is a boolean matrix of shape (Queries, Gallery) sorted by closeness
    sorted_matches = np.take_along_axis(match_matrix, sort_idx, axis=1)

    boot_results = []
    np.random.seed(random_state)

    print(f"-> Bootstrapping {mode} metrics from CSV ({m} iterations)...")

    for _ in tqdm(range(m)):
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        selected_row_indices = []
        for id_ in sampled_ids:
            selected_row_indices.extend(id_to_indices[id_])

        df_boot = df_full.iloc[selected_row_indices].copy()

        if mode == "closed":
            # Pass the pre-sorted matches to the logic function
            res = _calc_closed_logic_vectorized(sorted_matches[selected_indices])
            boot_results.append(res)
        else:
            # Open world is already relatively fast, but we pass pre-calculated masks
            res = _calc_open_logic_vectorized(dist_mat[selected_indices], 
                                              query_labels[selected_indices], 
                                              gallery_labels)
            boot_results.append(res)

    return _aggregate_bootstrap_results(boot_results, mode)

# -------- VECTORIZED INTERNAL HELPERS --------

def _calc_closed_logic_vectorized(matches):
    """
    Lightning fast vectorized CMC and mAP.
    matches: Boolean array (NumQueries, NumGallery) sorted by distance.
    """
    num_queries, num_gallery = matches.shape

    # --- CMC ---
    # Find index of first True in each row
    first_match_ranks = np.argmax(matches, axis=1)
    # Only count queries that actually have a match in gallery
    has_match = np.any(matches, axis=1)
    valid_ranks = first_match_ranks[has_match]
    
    cmc_counts = np.bincount(valid_ranks, minlength=num_gallery)
    cmc = np.cumsum(cmc_counts) / len(valid_ranks) if len(valid_ranks) > 0 else np.zeros(num_gallery)

    # --- mAP ---
    cum_matches = np.cumsum(matches, axis=1)
    precisions = cum_matches / np.arange(1, num_gallery + 1)
    # Average Precision is mean of precisions at the successful match positions
    ap = np.sum(precisions * matches, axis=1) / np.maximum(1, np.sum(matches, axis=1))
    
    return {
        "mAP": np.mean(ap[has_match]) if any(has_match) else 0,
        "cmc": cmc
    }

def _calc_open_logic_vectorized(dist_mat, q_labels, g_labels):
    """Vectorized open-world logic."""
    # Known vs Unknown
    known_mask = np.isin(q_labels, g_labels)
    unknown_mask = ~known_mask

    # Best matches
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)
    correct_match = (g_labels[best_idx] == q_labels)

    thresholds = np.linspace(0, 2, 500)
    
    # Broadcast comparison for DIR and FAR
    # Shape: (500, NumQueries)
    dirs_at_t = np.array([np.sum((best_dist[known_mask] < t) & correct_match[known_mask]) / np.sum(known_mask) 
                          if any(known_mask) else 0 for t in thresholds])
    fars_at_t = np.array([np.sum(best_dist[unknown_mask] < t) / np.sum(unknown_mask) 
                          if any(unknown_mask) else 0 for t in thresholds])

    res = {"dirs": dirs_at_t, "fars": fars_at_t}
    for target in [0.01, 0.05, 0.1]:
        idx = np.argmin(np.abs(fars_at_t - target))
        res[target] = dirs_at_t[idx]
    return res

def _aggregate_bootstrap_results(results, mode):
    # (Existing aggregation logic is fine, it just handles the dicts)
    if mode == "closed":
        maps = [r["mAP"] for r in results]
        cmcs = np.array([r["cmc"] for r in results])
        mean_cmc = np.mean(cmcs, axis=0)
        return {
            "mAP_mean": np.mean(maps), "mAP_std": np.std(maps),
            "cmc_mean": mean_cmc, "cmc_lower": np.percentile(cmcs, 2.5, axis=0),
            "cmc_upper": np.percentile(cmcs, 97.5, axis=0),
            "ranks": np.arange(1, len(mean_cmc) + 1)
        }
    else:
        all_dirs = np.array([r["dirs"] for r in results])
        all_fars = np.array([r["fars"] for r in results])
        return {
            "mean_fars": np.mean(all_fars, axis=0),
            "mean_dirs": np.mean(all_dirs, axis=0),
            "lower_dirs": np.percentile(all_dirs, 2.5, axis=0),
            "upper_dirs": np.percentile(all_dirs, 97.5, axis=0)
        }