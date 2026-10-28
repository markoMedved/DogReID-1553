import importlib.util
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load WarmupCosineDecayScheduler directly from RoVF-Meerkat-Reidentification
_ROVF_DIR = Path(__file__).resolve().parent.parent / "RoVF-Meerkat-Reidentification"
_SCHED_PATH = _ROVF_DIR / "lr_schedulers" / "cosine_decay.py"

_spec = importlib.util.spec_from_file_location("rovf_cosine_decay", str(_SCHED_PATH))
_sched_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sched_mod)
WarmupCosineDecayScheduler = _sched_mod.WarmupCosineDecayScheduler


def rovf_euclidean_distance(anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean distance measure from RoVF get_anchors/anchor_fn.py."""
    return torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))


@torch.no_grad()
def anchor_fn_hard_batched(
    model: nn.Module,
    pos_tokens: torch.Tensor,
    neg_tokens: torch.Tensor,
    device: torch.device,
):
    """
    Exact mathematical implementation of anchor_fn_hard from
    RoVF-Meerkat-Reidentification/get_anchors/anchor_fn.py, operating on
    stacked positive and negative clip tensors in a single batched GPU pass.

    Args:
        model: RoVFVideoModel
        pos_tokens: Tensor of shape (N_pos, T, ...) where N_pos >= 2
        neg_tokens: Tensor of shape (N_neg, T, ...) where N_neg >= 1
        device: torch.device

    Returns:
        (anchor, positive, negative) tensors each of shape (T, ...)
    """
    num_pos = pos_tokens.size(0)
    assert num_pos >= 2, "Positive pool must contain at least 2 clips."

    model.eval()
    batch_tens = torch.cat([pos_tokens, neg_tokens], dim=0).to(device, non_blocking=True)
    emb_tens = model(batch_tens)
    if isinstance(emb_tens, (tuple, list)):
        emb_tens = emb_tens[-1]

    positive_embeds = emb_tens[:num_pos]
    negative_embeds = emb_tens[num_pos:]

    # Pairwise distances between all positive embeddings
    dist_matrix = rovf_euclidean_distance(
        positive_embeds.unsqueeze(1), positive_embeds.unsqueeze(0)
    )
    dist_matrix.fill_diagonal_(float("-inf"))

    # Two positives with the largest distance
    max_distance, max_indices = torch.max(dist_matrix, dim=1)
    pos1 = torch.argmax(max_distance)
    pos2 = max_indices[pos1]

    # Distances between selected positives and all negatives
    dist_to_pos1 = rovf_euclidean_distance(
        negative_embeds, positive_embeds[pos1].unsqueeze(0)
    ).view(-1)
    dist_to_pos2 = rovf_euclidean_distance(
        negative_embeds, positive_embeds[pos2].unsqueeze(0)
    ).view(-1)

    min_neg_distance_for_pos1, neg_for_pos1 = torch.min(dist_to_pos1, dim=0)
    min_neg_distance_for_pos2, neg_for_pos2 = torch.min(dist_to_pos2, dim=0)

    if min_neg_distance_for_pos1 < min_neg_distance_for_pos2:
        anchor_idx, positive_idx = int(pos1.item()), int(pos2.item())
        negative_idx = int(neg_for_pos1.item())
    else:
        anchor_idx, positive_idx = int(pos2.item()), int(pos1.item())
        negative_idx = int(neg_for_pos2.item())

    return pos_tokens[anchor_idx], pos_tokens[positive_idx], neg_tokens[negative_idx]


def _extract_dataset_tokens(model, dataset, cfg, desc="Extracting DINOv2-small tokens"):
    """
    Pre-extract frozen DINOv2-small last_hidden_state token sequences (T, 257, 384)
    for all clips in a dataset (or load from disk cache if already extracted).
    """
    # Ensure deterministic uniform frame sampling (matches RoVF ReID.py np.linspace)
    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    orig_is_train = getattr(base_ds, "is_train", False)
    base_ds.is_train = False

    cache_dir = Path(cfg.project_root) / "checkpoints" / "rovf_token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_tag = "_masked" if getattr(cfg, "mask_dog", False) else ""
    dino_tag = str(getattr(cfg, "rovf_dino_model_name", "dinov2")).split("/")[-1].replace("-", "_")
    cache_file = cache_dir / f"{dino_tag}_tokens_T{cfg.clip_len}_{cfg.img_size[0]}x{cfg.img_size[1]}{mask_tag}.pt"

    cached_dict = {}
    if cache_file.exists():
        try:
            print(f"[RoVF] Loading cached DINOv2-small frame tokens from {cache_file}...")
            cached_dict = torch.load(cache_file, map_location="cpu")
            print(f"[RoVF] Loaded {len(cached_dict)} cached video clips.")
        except Exception as e:
            print(f"[RoVF] Warning: could not read cache ({e}), recomputing...")
            cached_dict = {}

    # Check which items in `dataset` are missing from cached_dict
    indices = dataset.indices if hasattr(dataset, "indices") else list(range(len(dataset)))
    needed_keys = []
    for idx in indices:
        row = base_ds.df.iloc[idx]
        key = f"{row['DOG_ID']}_{row['VIDEO_ID']}"
        if key not in cached_dict:
            needed_keys.append(key)

    if needed_keys:
        print(f"[RoVF] Extracting frozen DINOv2-small tokens for {len(needed_keys)} clips ({desc})...")
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        model.eval()
        with torch.no_grad():
            for videos, _labels, dog_ids, video_ids in tqdm(loader, desc=desc):
                videos = videos.to(cfg.device, non_blocking=True)
                tokens = model.extract_frame_tokens(videos).half().cpu()
                for j, (d_id, v_id) in enumerate(zip(dog_ids, video_ids)):
                    k = f"{d_id}_{v_id}"
                    cached_dict[k] = tokens[j]

        # Save updated cache atomically
        tmp_file = cache_file.with_suffix(f".tmp.{os.getpid()}.pt")
        try:
            torch.save(cached_dict, tmp_file)
            os.replace(tmp_file, cache_file)
            print(f"[RoVF] Saved {len(cached_dict)} clip token sequences to {cache_file}")
        except Exception as e:
            print(f"[RoVF] Warning: could not write token cache ({e})")
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    base_ds.is_train = orig_is_train

    # Build ordered tensors/lists for this dataset
    clip_tokens = []
    dog_ids_list = []
    video_ids_list = []
    labels_list = []

    for idx in indices:
        row = base_ds.df.iloc[idx]
        d_id, v_id = str(row["DOG_ID"]), str(row["VIDEO_ID"])
        k = f"{d_id}_{v_id}"
        clip_tokens.append(cached_dict[k])
        dog_ids_list.append(d_id)
        video_ids_list.append(v_id)
        labels_list.append(base_ds.labels[idx])

    return clip_tokens, dog_ids_list, video_ids_list, labels_list


@torch.no_grad()
def evaluate_rovf_tokens(
    model: nn.Module,
    q_tokens: list,
    q_dog_ids: list,
    q_video_ids: list,
    g_tokens: list,
    g_dog_ids: list,
    g_video_ids: list,
    device: torch.device,
    batch_size: int = 64,
):
    """Evaluate Rank-1, Rank-5, Rank-10, and mAP using RoVF's raw Euclidean distance."""
    model.eval()

    def _encode_all(token_list):
        embs = []
        for i in range(0, len(token_list), batch_size):
            batch = torch.stack(token_list[i : i + batch_size], dim=0).to(
                device, dtype=torch.float32, non_blocking=True
            )
            out = model(batch)
            embs.append(out.cpu())
        return torch.cat(embs, dim=0)

    q_emb = _encode_all(q_tokens)
    g_emb = _encode_all(g_tokens)

    dist_mat = torch.cdist(q_emb.to(device), g_emb.to(device), p=2).cpu().numpy()

    q_dogs = np.array(q_dog_ids)
    g_dogs = np.array(g_dog_ids)
    q_vids = np.array(q_video_ids)
    g_vids = np.array(g_video_ids)

    num_q = len(q_dogs)
    ranks = np.zeros(num_q, dtype=np.int32)
    aps = []
    valid_queries = 0

    for i in range(num_q):
        q_dog = q_dogs[i]
        q_vid = q_vids[i]
        dists = dist_mat[i].copy()

        # Exclude exact same video if present
        same_video = (g_dogs == q_dog) & (g_vids == q_vid)
        dists[same_video] = np.inf

        order = np.argsort(dists)
        matches = (g_dogs[order] == q_dog) & (~same_video[order])

        if not np.any(matches):
            continue

        valid_queries += 1
        first_match = int(np.argmax(matches))
        ranks[valid_queries - 1] = first_match

        # Average Precision
        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches[matches] / (np.where(matches)[0] + 1.0)
        aps.append(float(np.mean(precision_at_k)))

    if valid_queries == 0:
        return 0.0, 0.0, 0.0, 0.0

    ranks = ranks[:valid_queries]
    r1 = float(np.mean(ranks < 1))
    r5 = float(np.mean(ranks < 5))
    r10 = float(np.mean(ranks < 10))
    mAP = float(np.mean(aps))
    return r1, r5, r10, mAP


def train_rovf(cfg, model, train_loader, query_loader, gallery_loader):
    """
    Train RoVF-S-ST using the exact optimization, loss, and online hard triplet mining
    procedure from RoVF-Meerkat-Reidentification/training_functions/train.py.
    """
    device = cfg.device
    model.to(device)

    # 1. Pre-extract / load frozen DINOv2-small tokens for train, query, and gallery splits
    train_tokens, train_dogs, _train_vids, _train_labels = _extract_dataset_tokens(
        model, train_loader.dataset, cfg, desc="Train Split DINOv2-small Tokens"
    )
    q_tokens, q_dogs, q_vids, _ = _extract_dataset_tokens(
        model, query_loader.dataset, cfg, desc="Query Split DINOv2-small Tokens"
    )
    g_tokens, g_dogs, g_vids, _ = _extract_dataset_tokens(
        model, gallery_loader.dataset, cfg, desc="Gallery Split DINOv2-small Tokens"
    )

    # Organize training clip indices by dog_id
    dog_to_indices = defaultdict(list)
    for idx, d_id in enumerate(train_dogs):
        dog_to_indices[d_id].append(idx)

    valid_anchor_dogs = sorted([d_id for d_id, idxs in dog_to_indices.items() if len(idxs) >= 2])
    all_dog_ids = sorted(list(dog_to_indices.keys()))
    print(
        f"[RoVF] Training pool: {len(train_tokens)} clips across {len(all_dog_ids)} dogs "
        f"({len(valid_anchor_dogs)} dogs with >=2 clips for anchor sampling)."
    )

    # Pre-build negative index pools per dog for fast sampling
    all_indices_arr = np.arange(len(train_tokens))
    dog_neg_pool = {}
    for d_id in valid_anchor_dogs:
        pos_set = set(dog_to_indices[d_id])
        dog_neg_pool[d_id] = [i for i in all_indices_arr if i not in pos_set]

    # 2. Optimizer, Criterion, and Scheduler (Exact RoVF parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(getattr(cfg, "lr", 1e-4)),
        weight_decay=float(getattr(cfg, "weight_decay", 0.01)),
    )
    criterion = nn.TripletMarginLoss(
        margin=float(getattr(cfg, "margin", 1.0)),
        p=2,
    )

    episodes_per_epoch = int(getattr(cfg, "rovf_episodes_per_epoch", 5000))
    batch_size = int(getattr(cfg, "batch_size", 30))  # 30 clips = 10 triplets per step
    triplets_per_step = max(1, batch_size // 3)
    steps_per_epoch = episodes_per_epoch // triplets_per_step
    total_steps = cfg.epochs * steps_per_epoch

    warmup_steps = int(getattr(cfg, "rovf_warmup_steps", max(1, int(0.05 * total_steps))))
    decay_steps = int(getattr(cfg, "rovf_decay_steps", max(1, total_steps - warmup_steps)))
    start_lr = float(getattr(cfg, "rovf_start_lr", 0.001))
    max_lr = float(getattr(cfg, "rovf_max_lr", 0.005))
    end_lr = float(getattr(cfg, "rovf_end_lr", 0.0001))

    scheduler = WarmupCosineDecayScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        start_lr=start_lr,
        max_lr=max_lr,
        end_lr=end_lr,
    )

    K = int(getattr(cfg, "rovf_K", 20))
    eval_period = int(getattr(cfg, "eval_period", 1))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_config()

    print("\n" + "=" * 65)
    print("STARTING RoVF-S-ST TRAINING (Exact Original RoVF Hyperparameters)")
    print(f"  Backbone           : {cfg.rovf_dino_model_name} (frozen={cfg.rovf_freeze_image_model})")
    print(f"  Recurrent Head     : Perceiver-{cfg.rovf_perceiver_type.upper()} (skip_avg={cfg.rovf_is_append_avg_emb})")
    print(f"  Clip Length        : {cfg.clip_len} frames @ {cfg.img_size[0]}x{cfg.img_size[1]}")
    print(f"  Candidate Pool (K) : up to {K} pos & {K} neg clips per episode")
    print(f"  Episodes / Epoch   : {episodes_per_epoch} ({steps_per_epoch} optimizer steps/epoch, batch_size={batch_size})")
    print(f"  Total Epochs       : {cfg.epochs} ({total_steps} total optimizer steps)")
    print(f"  LR Schedule        : WarmupCosineDecay ({start_lr} -> {max_lr} over {warmup_steps} steps, cosine -> {end_lr} over {decay_steps} steps)")
    print(f"  Criterion          : TripletMarginLoss(margin={cfg.margin}, p=2)")
    print(f"  Output Directory   : {cfg.output_dir}")
    print("=" * 65 + "\n")

    best_mAP = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        cumulative_loss = 0.0
        effective_step = 0
        batched_list = []

        # Sample `episodes_per_epoch` anchor identities (matching RoVF ReID.py random.choice)
        episode_anchors = [random.choice(valid_anchor_dogs) for _ in range(episodes_per_epoch)]

        optimizer.zero_grad()

        for ep_idx, anchor_dog in enumerate(episode_anchors):
            pos_pool_idxs = dog_to_indices[anchor_dog]
            if len(pos_pool_idxs) > K:
                pos_pool_idxs = random.sample(pos_pool_idxs, K)

            neg_pool_all = dog_neg_pool[anchor_dog]
            neg_pool_idxs = random.sample(neg_pool_all, min(len(neg_pool_all), K))

            pos_cand = torch.stack([train_tokens[i] for i in pos_pool_idxs], dim=0).float()
            neg_cand = torch.stack([train_tokens[i] for i in neg_pool_idxs], dim=0).float()

            anchor_t, pos_t, neg_t = anchor_fn_hard_batched(model, pos_cand, neg_cand, device)
            batched_list.extend([anchor_t, pos_t, neg_t])

            if len(batched_list) >= batch_size or (ep_idx == len(episode_anchors) - 1 and len(batched_list) >= 3):
                model.train()
                batched_input = torch.stack(batched_list, dim=0).to(device, dtype=torch.float32, non_blocking=True)
                emb = model(batched_input)
                if isinstance(emb, (tuple, list)):
                    emb = emb[-1]

                emb_anchor = emb[0::3, :]
                emb_positive = emb[1::3, :]
                emb_negative = emb[2::3, :]

                loss = criterion(emb_anchor, emb_positive, emb_negative)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    cumulative_loss += float(loss.item())
                    effective_step += 1
                    global_step += 1

                batched_list = []

        avg_loss = cumulative_loss / max(1, effective_step)
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        if epoch % eval_period == 0 or epoch == cfg.epochs:
            r1, r5, r10, mAP = evaluate_rovf_tokens(
                model, q_tokens, q_dogs, q_vids, g_tokens, g_dogs, g_vids, device
            )
            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP

            ckpt_payload = {
                "model": model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "mAP": mAP,
                "rank1": r1,
                "rank5": r5,
                "rank10": r10,
            }
            torch.save(ckpt_payload, cfg.output_dir / "latest_model.pth")
            if epoch in (5, 10):
                torch.save(ckpt_payload, cfg.output_dir / f"checkpoint_epoch_{epoch}.pth")
            if is_best:
                torch.save(ckpt_payload, cfg.output_dir / "best_model.pth")

            best_str = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({epoch_time:.1f}s) | "
                f"Loss: {avg_loss:.4f} | LR: {cur_lr:.6f} | "
                f"R1: {r1*100:.2f}% | R5: {r5*100:.2f}% | R10: {r10*100:.2f}% | mAP: {mAP*100:.2f}%{best_str}"
            )
        else:
            print(
                f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({epoch_time:.1f}s) | "
                f"Loss: {avg_loss:.4f} | LR: {cur_lr:.6f}"
            )

    print(f"\n[RoVF] Training complete! Best Validation mAP: {best_mAP*100:.2f}%")
    print(f"[RoVF] Checkpoints saved to: {cfg.output_dir}")
