import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path


def extract_features_with_ids(model, dataloader, device):
    """
    Extract embeddings for every sample and attach a unique ID.

    Dataset returns:
    (clip, label, dog_id, video_id)

    We combine dog_id and video_id to build a unique identifier
    so later evaluation can recover the dog label.
    """
    model.eval()

    all_embeddings = []
    all_ids = []
    
    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader):

            # unpack dataset batch
            videos, labels, dog_ids, video_ids = batch
            
            # create unique sample id: "dogID_videoID"
            # example: "dog12_v3"
            item_ids = [f"{d}_{v}" for d, v in zip(dog_ids, video_ids)]
            
            videos = videos.to(device)

            # forward pass → embeddings
            embeddings = model(videos)
            
            # normalize so Euclidean distance behaves properly
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_ids.extend(item_ids)

    # concatenate all batches
    return torch.cat(all_embeddings), all_ids


def generate_distance_csv(model, query_loader, gallery_loader, cfg, filename="dist_matrix.csv"):
    """
    Build a pairwise distance matrix between query and gallery
    and save it as a CSV file.

    Rows = Query samples
    Columns = Gallery samples
    Values = Euclidean distances
    """
    
    # extract embeddings
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)
    
    print("-> Computing Distance Matrix...")

    # pairwise Euclidean distance
    # shape: [num_queries, num_gallery]
    dist_mat = torch.cdist(q_feat, g_feat, p=2).numpy()
    
    # create dataframe with gallery IDs as columns
    df = pd.DataFrame(dist_mat, columns=g_ids)

    # insert query IDs as first column
    df.insert(0, 'queryId', q_ids)
    
    # save CSV
    output_path = Path(cfg.output_dir) / filename
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"Distance CSV successfully created: {output_path}")

    return output_path


def calculate_metrics_from_csv(csv_path):
    """
    Compute standard ReID metrics (mAP, Rank-1, Rank-5, Rank-10)
    from a precomputed distance matrix CSV.
    """

    df = pd.read_csv(csv_path, sep=';')
    
    # first column = query IDs
    query_ids = df['queryId'].values

    # remaining columns = gallery IDs
    gallery_ids = df.columns[1:].values

    # extract distance matrix
    dist_mat = df.iloc[:, 1:].values
    
    # extract dog label from "dogID_videoID"
    get_dog_label = lambda x: str(x).split('_')[0]
    
    num_queries = len(query_ids)

    all_aps = []
    all_cmc = []

    print(f"-> Processing {num_queries} queries from {csv_path}...")

    for i in range(num_queries):

        q_label = get_dog_label(query_ids[i])
        
        # rank gallery by distance
        sort_idx = np.argsort(dist_mat[i])

        sorted_gallery_labels = [get_dog_label(gallery_ids[j]) for j in sort_idx]
        
        # correct matches
        matches = np.array([label == q_label for label in sorted_gallery_labels])
        
        # ----- CMC (rank metrics) -----

        if any(matches):

            first_match_rank = np.where(matches == True)[0][0]

            cmc = np.zeros(len(matches))
            cmc[first_match_rank:] = 1

            all_cmc.append(cmc)
        
        # ----- Average Precision -----

        num_rel = np.sum(matches)

        if num_rel > 0:

            cum_matches = np.cumsum(matches)

            precisions = cum_matches / (np.arange(len(matches)) + 1)

            ap = np.sum(precisions * matches) / num_rel

            all_aps.append(ap)

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
    Compute open-set metrics using a distance matrix.

    DIR = Detection Identification Rate
    FAR = False Accept Rate
    """

    df = pd.read_csv(csv_path, sep=';')
    
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values

    dist_mat = df.iloc[:, 1:].values 
    
    get_dog_label = lambda x: str(x).split('_')[0]
    
    q_labels = np.array([get_dog_label(i) for i in query_ids])
    g_labels = np.array([get_dog_label(i) for i in gallery_ids])
    
    # identify queries that exist in gallery
    known_mask = np.array([q in g_labels for q in q_labels])

    # queries that do NOT exist in gallery
    unknown_mask = ~known_mask
    
    if not any(unknown_mask):
        print("Warning: No 'Unknown' dogs found in query set. FAR will always be 0.")

    # best match per query
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)

    best_match_label = g_labels[best_idx]

    correct_match = (best_match_label == q_labels)
    
    if thresholds is None:
        thresholds = np.linspace(0, 2, 1000)
        
    dir_curve = []
    far_curve = []
    
    for t in thresholds:

        # correct identification of known dogs
        if any(known_mask):
            dir_val = np.sum((best_dist[known_mask] < t) & correct_match[known_mask]) / np.sum(known_mask)
        else:
            dir_val = 0
            
        # false acceptance of unknown dogs
        if any(unknown_mask):
            far_val = np.sum(best_dist[unknown_mask] < t) / np.sum(unknown_mask)
        else:
            far_val = 0
            
        dir_curve.append(dir_val)
        far_curve.append(far_val)
        
    dir_curve = np.array(dir_curve)
    far_curve = np.array(far_curve)

    print("\n" + "="*40)
    print("OPEN SET PERFORMANCE (DIR @ FAR)")
    print("-"*40)
    
    results = {}

    for target_far in [0.01, 0.05, 0.1]:

        idx = np.argmin(np.abs(far_curve - target_far))

        print(f"DIR @ {target_far*100:>4}% FAR: {dir_curve[idx]:.2%}")

        results[f"DIR@{target_far}"] = dir_curve[idx]
        
    print("="*40 + "\n")
    
    return thresholds, dir_curve, far_curve


def bootstrap_from_csv(csv_path, m=100, mode="closed", random_state=42):
    """
    Bootstrap evaluation by resampling query identities.

    This gives confidence intervals for metrics.
    """

    df_full = pd.read_csv(csv_path, sep=';')

    get_dog_label = lambda x: str(x).split('_')[0]
    
    query_labels = np.array([get_dog_label(i) for i in df_full['queryId'].values])

    unique_ids = np.unique(query_labels)
    
    id_to_indices = {id_: np.where(query_labels == id_)[0] for id_ in unique_ids}
    
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
            res = _calc_closed_logic(df_boot)
        else:
            res = _calc_open_logic(df_boot)

        boot_results.append(res)

    return _aggregate_bootstrap_results(boot_results, mode)