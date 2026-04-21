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

