import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

def extract_features_with_ids(model, dataloader, device):
    """
    Extracts features and returns them along with sample identifiers.
    Assumes Dataset __getitem__ returns (video, label, sample_id).
    """
    model.eval()
    all_embeddings = []
    all_ids = []
    
    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for videos, labels, item_ids in tqdm(dataloader):
            videos = videos.to(device)
            embeddings = model(videos)
            
            # Normalize for consistent Euclidean distance behavior
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            # item_ids should be the unique video filename or dog_name_index
            all_ids.extend(item_ids)
            
    return torch.cat(all_embeddings), all_ids

def generate_distance_csv(model, query_loader, gallery_loader, cfg, filename="dist_matrix.csv"):
    """
    Generates a CSV where:
    Rows = Query samples
    Cols = Gallery samples
    Values = Pairwise Distances
    Format: queryId;galleryId_1;galleryId_2;...
    """
    # 1. Get Embeddings and IDs
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)
    
    # 2. Compute Euclidean Distance Matrix (Query x Gallery)
    print("-> Computing Distance Matrix...")
    # Resulting shape: [num_queries, num_gallery]
    dist_mat = torch.cdist(q_feat, g_feat, p=2).numpy()
    
    # 3. Create DataFrame
    # Set the column names as the Gallery Sample IDs
    df = pd.DataFrame(dist_mat, columns=g_ids)
    
    # Insert the Query IDs at the start of the row
    df.insert(0, 'queryId', q_ids)
    
    # 4. Save to CSV with semicolon separator
    output_path = Path(cfg.output_dir) / filename
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"✅ Distance CSV successfully created: {output_path}")
    return output_path

def calculate_metrics_from_csv(csv_path):
    # 1. Load the distance matrix
    # Using sep=';' as defined in your CSV construction
    df = pd.read_csv(csv_path, sep=';')
    
    # queryId is the first column, the rest are galleryIds
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values  # Extract only the distance values
    
    # 2. Helper function to extract actual Dog ID from sample ID
    # Edit this if your IDs are formatted differently (e.g., 'dog001_v1' -> 'dog001')
    get_dog_label = lambda x: str(x).split('_')[0]
    
    num_queries = len(query_ids)
    all_aps = []
    all_cmc = []

    print(f"-> Processing {num_queries} queries from {csv_path}...")

    for i in range(num_queries):
        q_label = get_dog_label(query_ids[i])
        
        # Sort gallery indices by distance (ascending)
        sort_idx = np.argsort(dist_mat[i])
        sorted_gallery_labels = [get_dog_label(gallery_ids[j]) for j in sort_idx]
        
        # Create a boolean mask of matches
        matches = np.array([label == q_label for label in sorted_gallery_labels])
        
        # --- CMC Calculation (Rank-N) ---
        if any(matches):
            first_match_rank = np.where(matches == True)[0][0]
            cmc = np.zeros(len(matches))
            cmc[first_match_rank:] = 1
            all_cmc.append(cmc)
        
        # --- AP Calculation (mAP) ---
        num_rel = np.sum(matches)
        if num_rel > 0:
            # Cumulative matches: [0, 0, 1, 1, 2...]
            cum_matches = np.cumsum(matches)
            # Precision at each rank: [0, 0, 1/3, 2/4...]
            precisions = cum_matches / (np.arange(len(matches)) + 1)
            # AP = sum(precision at each correct rank) / total relevant
            ap = np.sum(precisions * matches) / num_rel
            all_aps.append(ap)

    # 3. Final Results summary
    mAP = np.mean(all_aps)
    cmc_avg = np.mean(all_cmc, axis=0)
    
    print("\n" + "="*40)
    print("PRECOMPUTED DISTANCE METRICS")
    print("-"*40)
    print(f"mAP      : {mAP:.2%}")
    print(f"Rank-1   : {cmc_avg[0]:.2%}")
    print(f"Rank-5   : {cmc_avg[4]:.2%}")
    print(f"Rank-10  : {cmc_avg[9]:.2%}")
    print("="*40 + "\n")

    return {"mAP": mAP, "Rank-1": cmc_avg[0]}



def calculate_open_set_metrics_from_csv(csv_path, thresholds=None):
    """
    Calculates DIR @ FAR levels from a precomputed distance CSV.
    Assumes queryId and gallery column names follow 'DogID_SampleID' format.
    """
    df = pd.read_csv(csv_path, sep=';')
    
    # 1. Setup IDs and Labels
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    # Distances (Rows = Queries, Cols = Gallery)
    dist_mat = df.iloc[:, 1:].values 
    
    get_dog_label = lambda x: str(x).split('_')[0]
    
    q_labels = np.array([get_dog_label(i) for i in query_ids])
    g_labels = np.array([get_dog_label(i) for i in gallery_ids])
    
    # 2. Identify Known vs Unknown Queries
    # A query is 'known' if its Dog ID exists in the gallery set
    known_mask = np.array([q in g_labels for q in q_labels])
    unknown_mask = ~known_mask
    
    if not any(unknown_mask):
        print("Warning: No 'Unknown' dogs found in query set. FAR will always be 0.")

    # 3. Precompute best matches (Minimum Distance)
    # For every query, find the smallest distance and the ID of that dog
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)
    best_match_label = g_labels[best_idx]
    
    # Logic for DIR: Distance < Threshold AND Label matches
    correct_match = (best_match_label == q_labels)
    
    # 4. Sweep Thresholds
    if thresholds is None:
        # Since your features are normalized, distances range from 0 to 2
        thresholds = np.linspace(0, 2, 1000)
        
    dir_curve = []
    far_curve = []
    
    for t in thresholds:
        # DIR: Fraction of KNOWN queries correctly identified with dist < t
        if any(known_mask):
            # Success = (is known) AND (dist < t) AND (correct dog)
            dir_val = np.sum((best_dist[known_mask] < t) & correct_match[known_mask]) / np.sum(known_mask)
        else:
            dir_val = 0
            
        # FAR: Fraction of UNKNOWN queries incorrectly accepted with dist < t
        if any(unknown_mask):
            # False Alarm = (is unknown) AND (any gallery dog is closer than t)
            far_val = np.sum(best_dist[unknown_mask] < t) / np.sum(unknown_mask)
        else:
            far_val = 0
            
        dir_curve.append(dir_val)
        far_curve.append(far_val)
        
    dir_curve = np.array(dir_curve)
    far_curve = np.array(far_curve)

    # 5. Extract specific operating points
    results = {}
    print("\n" + "="*40)
    print("OPEN SET PERFORMANCE (DIR @ FAR)")
    print("-"*40)
    
    for target_far in [0.01, 0.05, 0.1]: # 0.1%, 1%, 10%
        # Find index where far is closest to target
        idx = np.argmin(np.abs(far_curve - target_far))
        print(f"DIR @ {target_far*100:>4}% FAR: {dir_curve[idx]:.2%}")
        results[f"DIR@{target_far}"] = dir_curve[idx]
        
    print("="*40 + "\n")
    
    return thresholds, dir_curve, far_curve


def bootstrap_from_csv(csv_path, m=100, mode="closed", random_state=42):
    """
    Bootstraps metrics by resampling the rows of the precomputed distance CSV.
    """
    # 1. Load the full matrix once
    df_full = pd.read_csv(csv_path, sep=';')
    get_dog_label = lambda x: str(x).split('_')[0]
    
    # Extract labels for all queries (rows)
    query_labels = np.array([get_dog_label(i) for i in df_full['queryId'].values])
    unique_ids = np.unique(query_labels)
    
    # Map IDs to row indices in the dataframe
    id_to_indices = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}
    
    boot_results = []
    np.random.seed(random_state)

    print(f"-> Bootstrapping {mode} metrics from CSV ({m} iterations)...")
    
    for _ in tqdm(range(m)):
        # 2. Resample Dog IDs
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        
        # 3. Build the bootstrapped dataframe (resampled rows)
        selected_row_indices = []
        for id_ in sampled_ids:
            selected_row_indices.extend(id_to_indices[id_])
            
        # Create a temporary CSV-like DataFrame for this iteration
        df_boot = df_full.iloc[selected_row_indices].copy()
        
        # 4. Use your existing functions (modified slightly to take a DF instead of a path)
        if mode == "closed":
            # We call the logic from your calculate_metrics_from_csv
            res = _calc_closed_logic(df_boot)
            boot_results.append(res) # Store mAP and Rank-1
        else:
            # We call the logic from your calculate_open_set_metrics_from_csv
            res = _calc_open_logic(df_boot)
            boot_results.append(res) # Store DIR @ various FARs

    # 5. Aggregate Statistics
    return _aggregate_bootstrap_results(boot_results, mode)

# --- INTERNAL WRAPPERS FOR YOUR LOGIC ---

def _calc_closed_logic(df):
    """Your calculate_metrics_from_csv logic applied to a DataFrame."""
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values
    get_dog_label = lambda x: str(x).split('_')[0]
    
    all_aps = []
    all_r1 = []
    
    for i in range(len(query_ids)):
        q_label = get_dog_label(query_ids[i])
        sort_idx = np.argsort(dist_mat[i])
        matches = np.array([get_dog_label(gallery_ids[j]) == q_label for j in sort_idx])
        
        if any(matches):
            # Rank-1
            all_r1.append(1.0 if matches[0] else 0.0)
            # AP
            cum_matches = np.cumsum(matches)
            precisions = cum_matches / (np.arange(len(matches)) + 1)
            all_aps.append(np.sum(precisions * matches) / np.sum(matches))
            
    return {"mAP": np.mean(all_aps), "Rank-1": np.mean(all_r1)}

def _calc_open_logic(df):
    """Your calculate_open_set_metrics_from_csv logic applied to a DataFrame."""
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values
    get_dog_label = lambda x: str(x).split('_')[0]
    
    q_labels = np.array([get_dog_label(i) for i in query_ids])
    g_labels = np.array([get_dog_label(i) for i in gallery_ids])
    
    known_mask = np.array([q in g_labels for q in q_labels])
    unknown_mask = ~known_mask
    
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)
    correct_match = (g_labels[best_idx] == q_labels)
    
    thresholds = np.linspace(0, 2, 200) # Reduced for speed
    dir_at_far_points = {}
    
    # Calculate curves
    dirs, fars = [], []
    for t in thresholds:
        d = np.sum((best_dist[known_mask] < t) & correct_match[known_mask]) / np.sum(known_mask) if any(known_mask) else 0
        f = np.sum(best_dist[unknown_mask] < t) / np.sum(unknown_mask) if any(unknown_mask) else 0
        dirs.append(d)
        fars.append(f)
    
    # Extract specific targets (1%, 5%, 10%)
    for target in [0.01, 0.05, 0.1]:
        idx = np.argmin(np.abs(np.array(fars) - target))
        dir_at_far_points[target] = dirs[idx]
        
    return dir_at_far_points

def _aggregate_bootstrap_results(results, mode):
    if mode == "closed":
        maps = [r["mAP"] for r in results]
        r1s = [r["Rank-1"] for r in results]
        print(f"Bootstrap Results: mAP={np.mean(maps):.2%} ± {np.std(maps):.2%}, Rank-1={np.mean(r1s):.2%} ± {np.std(r1s):.2%}")
    else:
        for target in [0.01, 0.05, 0.1]:
            vals = [r[target] for r in results]
            print(f"Bootstrap DIR @ {target*100}% FAR: {np.mean(vals):.2%} [CI: {np.percentile(vals, 2.5):.2%} - {np.percentile(vals, 97.5):.2%}]")