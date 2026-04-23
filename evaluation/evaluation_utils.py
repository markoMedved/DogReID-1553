import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path


def extract_features_with_ids(model, dataloader, device):
    """
    Extract embeddings from the model while keeping track of sample IDs.

    The DOGVideoREIDDataset returns:
        clip      : video tensor (B, T, C, H, W)
        label     : class label
        dog_id    : dog identity
        video_id  : specific video identifier

    We combine dog_id and video_id to create a unique identifier for each sample.
    """
    model.eval()

    # store embeddings and IDs for all samples
    all_embeddings = []
    all_ids = []

    print(f"-> Extracting features for {len(dataloader.dataset)} samples...")

    # no gradients needed during feature extraction
    with torch.no_grad():
        for batch in tqdm(dataloader):

            # unpack dataset output
            videos, labels, dog_ids, video_ids = batch

            # create unique sample IDs like: "dog23_vid4"
            # this allows easy recovery of dog identity later
            item_ids = [f"{d}_{v}" for d, v in zip(dog_ids, video_ids)]

            # move video batch to GPU / device
            videos = videos.to(device)

            # forward pass through the model
            embeddings = model(videos)

            # normalize embeddings (important for stable distance comparisons)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # store results
            all_embeddings.append(embeddings.cpu())
            all_ids.extend(item_ids)

    # concatenate all batch embeddings into a single tensor
    return torch.cat(all_embeddings), all_ids


def generate_distance_csv(model, query_loader, gallery_loader, cfg, filename="dist_matrix.csv"):
    """
    Generate a distance matrix CSV for evaluation.

    Output format:
        rows    = query samples
        columns = gallery samples
        values  = pairwise Euclidean distances

    CSV structure:
        queryId ; gallery1 ; gallery2 ; gallery3 ...
    """

    # extract embeddings for queries and gallery
    q_feat, q_ids = extract_features_with_ids(model, query_loader, cfg.device)
    g_feat, g_ids = extract_features_with_ids(model, gallery_loader, cfg.device)

    print("-> Computing Distance Matrix...")

    # compute pairwise Euclidean distances
    # resulting shape: (num_queries, num_gallery)
    dist_mat = torch.cdist(q_feat, g_feat, p=2).numpy()

    # build dataframe where columns correspond to gallery IDs
    df = pd.DataFrame(dist_mat, columns=g_ids)

    # insert query IDs as first column
    df.insert(0, 'queryId', q_ids)

    # save to CSV (semicolon used to avoid conflicts with IDs)
    output_path = Path(cfg.output_dir) / filename
    df.to_csv(output_path, sep=';', index=False)

    print(f"Distance CSV successfully created: {output_path}")

    return output_path


def calculate_metrics_from_csv(csv_path):
    """
    Compute closed-set ReID metrics from a precomputed distance matrix.

    Metrics:
        mAP   (mean Average Precision)
        Rank-1
        Rank-5
        Rank-10
    """

    # load CSV
    df = pd.read_csv(csv_path, sep=';')

    # first column = query IDs
    query_ids = df['queryId'].values

    # remaining columns = gallery IDs
    gallery_ids = df.columns[1:].values

    # distance matrix values
    dist_mat = df.iloc[:, 1:].values

    # helper: extract dog identity from ID string
    # example: "dog12_video3" -> "dog12"
    get_dog_label = lambda x: str(x).split('_')[0]

    num_queries = len(query_ids)

    # store per-query metrics
    all_aps = []
    all_cmc = []

    print(f"-> Processing {num_queries} queries from {csv_path}...")

    for i in range(num_queries):

        # identity of current query
        q_label = get_dog_label(query_ids[i])

        # sort gallery indices by distance (smallest first)
        sort_idx = np.argsort(dist_mat[i])

        # get gallery labels in sorted order
        sorted_gallery_labels = [
            get_dog_label(gallery_ids[j]) for j in sort_idx
        ]

        # boolean match vector
        matches = np.array([label == q_label for label in sorted_gallery_labels])

        # ---------- CMC (Rank-N) ----------
        if any(matches):

            # index of first correct match
            first_match_rank = np.where(matches == True)[0][0]

            # build CMC vector
            cmc = np.zeros(len(matches))
            cmc[first_match_rank:] = 1

            all_cmc.append(cmc)

        # ---------- AP calculation ----------
        num_rel = np.sum(matches)

        if num_rel > 0:

            # cumulative number of matches
            cum_matches = np.cumsum(matches)

            # precision at each rank
            precisions = cum_matches / (np.arange(len(matches)) + 1)

            # average precision
            ap = np.sum(precisions * matches) / num_rel

            all_aps.append(ap)

    # aggregate final metrics
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
    Compute open-set identification metrics.

    Evaluates:
        DIR (Detection and Identification Rate)
        FAR (False Accept Rate)

    DIR@FAR values are reported at FAR levels:
        1%, 5%, 10%
    """

    df = pd.read_csv(csv_path, sep=';')

    # query and gallery IDs
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values

    # distance matrix
    dist_mat = df.iloc[:, 1:].values

    get_dog_label = lambda x: str(x).split('_')[0]

    # extract labels
    q_labels = np.array([get_dog_label(i) for i in query_ids])
    g_labels = np.array([get_dog_label(i) for i in gallery_ids])

    # determine which queries have known identities
    known_mask = np.array([q in g_labels for q in q_labels])
    unknown_mask = ~known_mask

    if not any(unknown_mask):
        print("Warning: No unknown dogs in query set.")

    # best gallery match for each query
    best_dist = np.min(dist_mat, axis=1)
    best_idx = np.argmin(dist_mat, axis=1)

    best_match_label = g_labels[best_idx]

    # correct match indicator
    correct_match = (best_match_label == q_labels)

    # threshold range for evaluation
    if thresholds is None:
        thresholds = np.linspace(0, 2, 1000)

    dir_curve = []
    far_curve = []

    for t in thresholds:

        # Detection & Identification Rate
        if any(known_mask):
            dir_val = np.sum(
                (best_dist[known_mask] < t) &
                correct_match[known_mask]
            ) / np.sum(known_mask)
        else:
            dir_val = 0

        # False Accept Rate
        if any(unknown_mask):
            far_val = np.sum(
                best_dist[unknown_mask] < t
            ) / np.sum(unknown_mask)
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
    Bootstrap evaluation to estimate metric variability.

    Procedure:
        1. Sample dog IDs with replacement
        2. Collect all query rows belonging to those IDs
        3. Recompute metrics
        4. Repeat m times
    """

    df_full = pd.read_csv(csv_path, sep=';')

    get_dog_label = lambda x: str(x).split('_')[0]

    # extract dog labels for each query row
    query_labels = np.array([
        get_dog_label(i) for i in df_full['queryId'].values
    ])

    # unique identities
    unique_ids = np.unique(query_labels)

    # map each dog ID to its row indices
    id_to_indices = {
        id_: np.where(query_labels == id_)[0]
        for id_ in unique_ids
    }

    boot_results = []

    np.random.seed(random_state)

    print(f"-> Bootstrapping {mode} metrics from CSV ({m} iterations)...")

    for _ in tqdm(range(m)):

        # sample identities with replacement
        sampled_ids = np.random.choice(
            unique_ids,
            size=len(unique_ids),
            replace=True
        )

        # collect rows corresponding to sampled IDs
        selected_row_indices = []

        for id_ in sampled_ids:
            selected_row_indices.extend(id_to_indices[id_])

        # create bootstrap dataset
        df_boot = df_full.iloc[selected_row_indices].copy()

        # compute metrics
        if mode == "closed":

            res = _calc_closed_logic(df_boot)
            boot_results.append(res)

        else:

            res = _calc_open_logic(df_boot)
            boot_results.append(res)

    # summarize bootstrap statistics
    return _aggregate_bootstrap_results(boot_results, mode)


# -------- INTERNAL HELPERS --------


def _calc_closed_logic(df):
    """
    Closed-set metric computation applied to a dataframe.
    """
    query_ids = df['queryId'].values
    gallery_ids = df.columns[1:].values
    dist_mat = df.iloc[:, 1:].values

    get_dog_label = lambda x: str(x).split('_')[0]

    all_aps = []
    all_r1 = []

    for i in range(len(query_ids)):

        q_label = get_dog_label(query_ids[i])

        sort_idx = np.argsort(dist_mat[i])

        matches = np.array([
            get_dog_label(gallery_ids[j]) == q_label
            for j in sort_idx
        ])

        if any(matches):

            # Rank-1 accuracy
            all_r1.append(1.0 if matches[0] else 0.0)

            # AP calculation
            cum_matches = np.cumsum(matches)
            precisions = cum_matches / (np.arange(len(matches)) + 1)

            all_aps.append(
                np.sum(precisions * matches) / np.sum(matches)
            )

    return {
        "mAP": np.mean(all_aps),
        "Rank-1": np.mean(all_r1)
    }


def _calc_open_logic(df):
    """
    Open-set metric computation applied to a dataframe.
    """

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

    thresholds = np.linspace(0, 2, 200)

    dirs, fars = [], []

    for t in thresholds:

        d = (
            np.sum((best_dist[known_mask] < t) & correct_match[known_mask])
            / np.sum(known_mask)
            if any(known_mask) else 0
        )

        f = (
            np.sum(best_dist[unknown_mask] < t)
            / np.sum(unknown_mask)
            if any(unknown_mask) else 0
        )

        dirs.append(d)
        fars.append(f)

    dir_at_far_points = {}

    for target in [0.01, 0.05, 0.1]:

        idx = np.argmin(np.abs(np.array(fars) - target))

        dir_at_far_points[target] = dirs[idx]

    return dir_at_far_points


def _aggregate_bootstrap_results(results, mode):
    """
    Summarize bootstrap metrics.
    """

    if mode == "closed":

        maps = [r["mAP"] for r in results]
        r1s = [r["Rank-1"] for r in results]

        print(
            f"Bootstrap Results: "
            f"mAP={np.mean(maps):.2%} ± {np.std(maps):.2%}, "
            f"Rank-1={np.mean(r1s):.2%} ± {np.std(r1s):.2%}"
        )

    else:

        for target in [0.01, 0.05, 0.1]:

            vals = [r[target] for r in results]

            print(
                f"Bootstrap DIR @ {target*100}% FAR: "
                f"{np.mean(vals):.2%} "
                f"[CI: {np.percentile(vals, 2.5):.2%} - {np.percentile(vals, 97.5):.2%}]"
            )