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

def generate_distance_csv(model, query_loader, gallery_loader, cfg, filename="dist_matrix.csv"):
    # --- Extract Features ---
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)

    print("-> Computing Distance Matrix...")
    # Compute L2 distance matrix between query and gallery sets
    dist_mat = torch.cdist(q_feat, g_feat, p=2).numpy()

    # --- Format as DataFrame ---
    df = pd.DataFrame(dist_mat, columns=g_ids)
    df.insert(0, 'queryId', q_ids)

    # --- Save to CSV ---
    output_path = Path(cfg.output_dir) / filename
    # Fixed: writing with a comma
    df.to_csv(output_path, sep=',', index=False)
    print(f"Distance CSV successfully created: {output_path}")
    return output_path

def bootstrap_from_csv(csv_path, m=100, mode="closed", random_state=42):
    # --- Load Distance Matrix ---
    # Fixed: reading with a comma
    df_full = pd.read_csv(csv_path, sep=',')
    
    # Helper to extract dog ID from the combined string format
    get_dog_label = lambda x: str(x).split('_')[0]

    # --- Map Identities to Rows ---
    query_labels = np.array([get_dog_label(i) for i in df_full['queryId'].values])
    unique_ids = np.unique(query_labels)
    id_to_indices = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}

    boot_results = []
    np.random.seed(random_state)

    print(f"-> Bootstrapping {mode} metrics from CSV ({m} iterations)...")

    # --- Bootstrap Sampling Loop ---
    for _ in tqdm(range(m)):
        # Sample identities with replacement
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        
        selected_row_indices = []
        for id_ in sampled_ids:
            selected_row_indices.extend(id_to_indices[id_])

        # Create resampled DataFrame
        df_boot = df_full.iloc[selected_row_indices].copy()

        # --- Evaluate Resampled Data ---
        if mode == "closed":
            res = _calc_closed_logic(df_boot)
            boot_results.append(res)
        else:
            res = _calc_open_logic(df_boot)
            boot_results.append(res)

    return _aggregate_bootstrap_results(boot_results, mode)

# -------- INTERNAL HELPERS --------

def _calc_closed_logic(df):
    """
    Computes mAP and the full CMC vector for a single bootstrap sample.
    """
    # --- Parse DataFrame ---
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values
    
    get_dog_label = lambda x: str(x).split('_')[0]

    all_aps = []
    all_cmcs = []

    # --- Calculate Metrics per Query ---
    for i in range(len(query_ids)):
        q_label = get_dog_label(query_ids[i])
        
        # Sort gallery indices by closest distance
        sort_idx = np.argsort(dist_mat[i])
        
        # Boolean array indicating correct matches
        matches = np.array([get_dog_label(gallery_ids[j]) == q_label for j in sort_idx])

        if any(matches):
            # --- CMC Vector Calculation ---
            first_match_rank = np.where(matches == True)[0][0]
            cmc = np.zeros(len(matches))
            cmc[first_match_rank:] = 1
            all_cmcs.append(cmc)

            # --- Average Precision (AP) Calculation ---
            cum_matches = np.cumsum(matches)
            precisions = cum_matches / (np.arange(len(matches)) + 1)
            all_aps.append(np.sum(precisions * matches) / np.sum(matches))

    return {
        "mAP": np.mean(all_aps) if all_aps else 0,
        "cmc": np.mean(all_cmcs, axis=0) if all_cmcs else np.zeros(len(gallery_ids))
    }

def _calc_open_logic(df):
    # --- Parse DataFrame ---
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values
    
    get_dog_label = lambda x: str(x).split('_')[0]

    # --- Extract Labels ---
    q_labels = np.array([get_dog_label(i) for i in query_ids])
    g_labels = np.array([get_dog_label(i) for i in gallery_ids])

    # --- Known vs Unknown Split ---
    known_mask = np.array([q in g_labels for q in q_labels])
    unknown_mask = ~known_mask

    # --- Find Best Matches ---
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)
    correct_match = (g_labels[best_idx] == q_labels)

    # Standardize 500 points for a smooth curve
    thresholds = np.linspace(0, 2, 500)
    dirs, fars = [], []

    # --- Calculate DIR and FAR across Thresholds ---
    for t in thresholds:
        d = np.sum((best_dist[known_mask] < t) & correct_match[known_mask]) / np.sum(known_mask) if any(known_mask) else 0
        f = np.sum(best_dist[unknown_mask] < t) / np.sum(unknown_mask) if any(unknown_mask) else 0
        dirs.append(d)
        fars.append(f)

    # --- Package Results ---
    # Return the full arrays for plotting, plus the specific points for the table
    res = {"dirs": np.array(dirs), "fars": np.array(fars)}
    
    for target in [0.01, 0.05, 0.1]:
        idx = np.argmin(np.abs(np.array(fars) - target))
        res[target] = dirs[idx]
        
    return res

def _aggregate_bootstrap_results(results, mode):
    # --- Aggregate Closed-Set Metrics ---
    if mode == "closed":
        maps = [r["mAP"] for r in results]
        cmcs = np.array([r["cmc"] for r in results]) # Shape: (m, num_gallery)

        # Compute mean and 95% confidence intervals
        mean_cmc = np.mean(cmcs, axis=0)
        lower_cmc = np.percentile(cmcs, 2.5, axis=0)
        upper_cmc = np.percentile(cmcs, 97.5, axis=0)

        print(f"\nBootstrap Results (Closed-Set):")
        print(f"mAP: {np.mean(maps):.2%} ± {np.std(maps):.2%}")
        print(f"Rank-1: {mean_cmc[0]:.2%} [{lower_cmc[0]:.2%} - {upper_cmc[0]:.2%}]")

        return {
            "mAP_mean": np.mean(maps),
            "mAP_std": np.std(maps),
            "cmc_mean": mean_cmc,
            "cmc_lower": lower_cmc,
            "cmc_upper": upper_cmc,
            "ranks": np.arange(1, len(mean_cmc) + 1)
        }
        
    # --- Aggregate Open-Set Metrics ---
    else:
        # Extract the full curves from all bootstrap iterations
        all_dirs = np.array([r["dirs"] for r in results]) # Shape (m, 500)
        all_fars = np.array([r["fars"] for r in results]) # Shape (m, 500)

        # Compute Mean and 95% Confidence Intervals for the curve
        mean_dirs = np.mean(all_dirs, axis=0)
        lower_dirs = np.percentile(all_dirs, 2.5, axis=0)
        upper_dirs = np.percentile(all_dirs, 97.5, axis=0)
        
        # We assume fars are roughly consistent across iterations because thresholds are fixed
        mean_fars = np.mean(all_fars, axis=0)

        print(f"\nBootstrap Results (Open-Set):")
        for target in [0.01, 0.05, 0.1]:
            vals = [r[target] for r in results]
            print(f"DIR @ {target:.0%} FAR: {np.mean(vals):.2%} ± {np.std(vals):.2%}")

        return {
            "mean_fars": mean_fars,
            "mean_dirs": mean_dirs,
            "lower_dirs": lower_dirs,
            "upper_dirs": upper_dirs,
            "targets": {t: np.mean([r[t] for r in results]) for t in [0.01, 0.05, 0.1]}
        }