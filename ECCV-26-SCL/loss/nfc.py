from collections import deque

import torch
import torch.nn.functional as F


class NFCQueue:
    def __init__(self, queue_size=4096, feature_dim=768):
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.features = deque(maxlen=self.queue_size)
        self.pids = deque(maxlen=self.queue_size)
        self.species = deque(maxlen=self.queue_size)

    def enqueue(self, features, pids, spes=None):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if pids.dim() == 0:
            pids = pids.unsqueeze(0)
        if spes is not None and spes.dim() == 0:
            spes = spes.unsqueeze(0)

        for i, (feat, pid) in enumerate(zip(features, pids)):
            self.features.append(feat.detach().cpu())
            self.pids.append(int(pid.item()))
            if spes is not None:
                self.species.append(int(spes[i].item()))

    def get_features(self, device):
        if not self.features:
            return torch.empty(0, self.feature_dim, device=device)
        return torch.stack(list(self.features)).to(device)

    def get_pids(self, device):
        if not self.pids:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(list(self.pids), dtype=torch.long, device=device)

    def get_species(self, device):
        if not self.species:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(list(self.species), dtype=torch.long, device=device)


def pairwise_distance(query_features, gallery_features):
    x = query_features.view(query_features.size(0), -1)
    y = gallery_features.view(gallery_features.size(0), -1)
    dist = (
        x.pow(2).sum(dim=1, keepdim=True)
        + y.pow(2).sum(dim=1, keepdim=True).t()
        - 2 * x @ y.t()
    )
    return dist.clamp_min(1e-12)


def find_mutual_neighbors(features, k1=8, k2=8):
    if features.size(0) <= 1:
        return [[] for _ in range(features.size(0))]
    k1 = min(int(k1), features.size(0) - 1)
    k2 = min(int(k2), k1)
    dist = pairwise_distance(features, features)
    dist.fill_diagonal_(float("inf"))
    rank = dist.topk(k1, largest=False).indices
    rank_k2 = rank[:, :k2]

    num_samples = rank.size(0)
    indices = torch.arange(num_samples, device=features.device).view(num_samples, 1, 1)
    membership = (indices == rank_k2.unsqueeze(0)).any(dim=2)

    mutual_neighbors = []
    rank_cpu = rank.cpu()
    membership_cpu = membership.cpu()
    for i in range(num_samples):
        neighbors_i = rank_cpu[i]
        mutual_mask = membership_cpu[i, neighbors_i]
        mutual_neighbors.append(neighbors_i[mutual_mask].tolist())
    return mutual_neighbors


def _mean_center(features, fallback, neighbors):
    if neighbors:
        return features[neighbors].mean(dim=0)
    return fallback


def compute_centers(features, fallback_features, neighbor_lists):
    centers = [
        _mean_center(features, fallback_features[i], neighbors)
        for i, neighbors in enumerate(neighbor_lists)
    ]
    return torch.stack(centers)


def find_same_and_cross_species_neighbors(
    features,
    speids,
    query_speids,
    k_same=8,
    k_cross_k1=4,
    k_cross_k2=4,
):
    batch_size = query_speids.size(0)
    dist_full = pairwise_distance(features, features)
    dist_query = dist_full[:batch_size]
    same_neighbors = []
    cross_neighbors = []

    for i in range(batch_size):
        query_species = query_speids[i].item()

        same_mask = speids == query_species
        same_mask[i] = False
        same_dist = dist_query[i].clone()
        same_dist[~same_mask] = float("inf")
        same_count = int(same_mask.sum().item())
        if same_count > 0:
            k = min(k_same, same_count)
            same_neighbors.append(torch.topk(same_dist, k=k, largest=False).indices.tolist())
        else:
            same_neighbors.append([])

        cross_mask = speids != query_species
        cross_mask[i] = False
        cross_dist = dist_query[i].clone()
        cross_dist[~cross_mask] = float("inf")
        cross_count = int(cross_mask.sum().item())
        if cross_count > 0:
            k = min(k_cross_k1, cross_count)
            cross_candidates = torch.topk(cross_dist, k=k, largest=False).indices
        else:
            cross_candidates = torch.empty(0, dtype=torch.long, device=features.device)

        mutual_cross = []
        for candidate in cross_candidates:
            j = int(candidate.item())
            j_species = speids[j].item()
            j_cross_mask = speids != j_species
            j_cross_mask[j] = False
            j_dist = dist_full[j].clone()
            j_dist[~j_cross_mask] = float("inf")
            j_count = int(j_cross_mask.sum().item())
            if j_count == 0:
                continue
            k = min(k_cross_k2, j_count)
            if (torch.topk(j_dist, k=k, largest=False).indices == i).any():
                mutual_cross.append(j)
        cross_neighbors.append(mutual_cross)

    return same_neighbors, cross_neighbors


class NFCTrainer:
    def __init__(
        self,
        queue_size=4096,
        k1=8,
        k2=8,
        temperature=0.07,
        feature_dim=768,
        same_species_weight=1.0,
        enable_cross_species=True,
        cross_species_k1=4,
        cross_species_k2=4,
        cross_species_weight=3.0,
        cross_species_margin=1.0,
        cross_species_metric="cosine",
    ):
        if cross_species_metric not in {"cosine", "euclidean"}:
            raise ValueError("cross_species_metric must be 'cosine' or 'euclidean'.")
        self.queue = NFCQueue(queue_size, feature_dim)
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.temperature = float(temperature)
        self.feature_dim = int(feature_dim)
        self.same_species_weight = float(same_species_weight)
        self.enable_cross_species = bool(enable_cross_species)
        self.cross_species_k1 = int(cross_species_k1)
        self.cross_species_k2 = int(cross_species_k2)
        self.cross_species_weight = float(cross_species_weight)
        self.cross_species_margin = float(cross_species_margin)
        self.cross_species_metric = cross_species_metric

    def update_queue(self, teacher_features, pids, spes=None):
        self.queue.enqueue(teacher_features, pids, spes)

    def compute_nfc_loss(self, student_features, teacher_features, pids, device, spes=None):
        batch_size = student_features.size(0)
        queue_features = self.queue.get_features(device)
        queue_pids = self.queue.get_pids(device)
        queue_spes = self.queue.get_species(device)

        if queue_features.numel() == 0:
            all_features = teacher_features
            all_pids = pids
            all_spes = spes
        else:
            all_features = torch.cat([teacher_features, queue_features], dim=0)
            all_pids = torch.cat([pids, queue_pids], dim=0)
            all_spes = torch.cat([spes, queue_spes], dim=0) if spes is not None and queue_spes.numel() > 0 else None

        cross_centers = None
        if self.enable_cross_species and spes is not None and all_spes is not None:
            same_neighbors, cross_neighbors = find_same_and_cross_species_neighbors(
                all_features,
                all_spes,
                spes,
                k_same=self.k1,
                k_cross_k1=self.cross_species_k1,
                k_cross_k2=self.cross_species_k2,
            )
            batch_centers = compute_centers(all_features, teacher_features, same_neighbors).detach()
            cross_centers = compute_centers(all_features, teacher_features, cross_neighbors).detach()
            stats = self._compute_species_statistics(same_neighbors, all_pids, all_spes, pids, spes)
            stats["cross_species_neighbors_count"] = sum(len(x) for x in cross_neighbors)
        else:
            mutual_neighbors = find_mutual_neighbors(all_features, self.k1, self.k2)
            batch_neighbors = mutual_neighbors[:batch_size]
            batch_centers = compute_centers(all_features, teacher_features, batch_neighbors).detach()
            stats = self._compute_species_statistics(batch_neighbors, all_pids, all_spes, pids, spes)
            stats["cross_species_neighbors_count"] = 0

        student_norm = F.normalize(student_features, p=2, dim=1)
        center_norm = F.normalize(batch_centers, p=2, dim=1)
        same_similarity = torch.sum(student_norm * center_norm, dim=1)
        same_loss = 1.0 - same_similarity.mean()
        nfc_loss = self.same_species_weight * same_loss

        if self.enable_cross_species and cross_centers is not None:
            if self.cross_species_metric == "cosine":
                cross_norm = F.normalize(cross_centers, p=2, dim=1)
                cross_similarity = torch.sum(student_norm * cross_norm, dim=1)
                cross_loss = torch.clamp(self.cross_species_margin - cross_similarity, min=0).mean()
                stats["cross_species_similarity"] = float(cross_similarity.mean().detach().item())
            else:
                cross_distance = torch.norm(student_features - cross_centers, p=2, dim=1)
                cross_loss = torch.clamp(cross_distance - self.cross_species_margin, min=0).mean()
                stats["cross_species_distance"] = float(cross_distance.mean().detach().item())
            nfc_loss = nfc_loss + self.cross_species_weight * cross_loss
            stats["cross_species_loss"] = float(cross_loss.detach().item())

        stats["same_species_loss"] = float(same_loss.detach().item())
        return nfc_loss, stats

    def _compute_species_statistics(self, neighbor_lists, all_pids, all_spes, batch_pids, batch_spes):
        stats = {
            "total_samples": len(neighbor_lists),
            "total_neighbors": 0,
            "same_species_neighbors": 0,
            "same_id_neighbors": 0,
            "same_species_ratio": 0.0,
            "same_id_ratio": 0.0,
            "avg_neighbors_per_sample": 0.0,
            "samples_with_neighbors": 0,
        }
        if all_spes is None or batch_spes is None:
            return stats

        batch_size = len(batch_pids)
        for i in range(batch_size):
            neighbors = neighbor_lists[i]
            if not neighbors:
                continue
            stats["samples_with_neighbors"] += 1
            stats["total_neighbors"] += len(neighbors)
            current_spe = batch_spes[i].item()
            current_pid = batch_pids[i].item()
            for neighbor_idx in neighbors:
                if all_spes[neighbor_idx].item() == current_spe:
                    stats["same_species_neighbors"] += 1
                if all_pids[neighbor_idx].item() == current_pid:
                    stats["same_id_neighbors"] += 1

        if stats["total_neighbors"] > 0:
            stats["same_species_ratio"] = stats["same_species_neighbors"] / stats["total_neighbors"]
            stats["same_id_ratio"] = stats["same_id_neighbors"] / stats["total_neighbors"]
        if stats["samples_with_neighbors"] > 0:
            stats["avg_neighbors_per_sample"] = stats["total_neighbors"] / stats["samples_with_neighbors"]
        return stats


class EMAModel:
    def __init__(self, model, momentum=0.993):
        self.model = model
        self.momentum = float(momentum)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    (1.0 - self.momentum) * param.detach()
                    + self.momentum * self.shadow[name]
                ).clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
