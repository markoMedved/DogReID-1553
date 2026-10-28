import torch
import numpy as np

from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids=None, g_camids=None, max_rank=50, cross_camera=False):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    
    Args:
        distmat: distance matrix (num_query x num_gallery)
        q_pids: query person IDs
        g_pids: gallery person IDs
        q_camids: query camera IDs (optional, required if cross_camera=True)
        g_camids: gallery camera IDs (optional, required if cross_camera=True)
        max_rank: maximum rank for CMC
        cross_camera: if True, remove same-camera matches (cross-camera retrieval)
    """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    np.fill_diagonal(distmat, np.inf)
    indices = np.argsort(distmat, axis=1)

    indices = indices[:, :-1]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []

    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx] if q_camids is not None else None

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        
        # 根据 cross_camera 参数决定是否过滤同摄像头图像
        if cross_camera and q_camid is not None and g_camids is not None:
            # 启用跨摄像头检索：移除同一个人ID且同一个摄像头ID的图像
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            # 默认行为：不移除（保持原有逻辑）
            remove = (g_pids[order] == q_pid) & False
        
        keep = np.invert(remove)
        matches = (g_pids[order] == q_pid).astype(np.int32)
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[keep]
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)

        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        # 确保所有CMC长度一致，填充0以达到max_rank
        cmc_padded = np.zeros(max_rank)
        cmc_len = min(len(cmc), max_rank)
        cmc_padded[:cmc_len] = cmc[:cmc_len]
        if cmc_len > 0:
            # 如果cmc长度小于max_rank，用最后一个值填充
            cmc_padded[cmc_len:] = cmc[cmc_len-1] if cmc_len < max_rank else cmc[max_rank-1]
        
        all_cmc.append(cmc_padded)
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP


class R1_mAP_eval:
    def __init__(self, max_rank=50, feat_norm=True, reranking=False, cross_camera=False):
        super(R1_mAP_eval, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.cross_camera = cross_camera

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        if len(output) == 2:
            # 兼容旧版：只有 feat 和 pid
            feat, pid = output
            self.feats.append(feat.cpu())
            self.pids.extend(np.asarray(pid))
        elif len(output) == 3:
            # 新版：feat, pid, camid
            feat, pid, camid = output
            self.feats.append(feat.cpu())
            self.pids.extend(np.asarray(pid))
            self.camids.extend(np.asarray(camid))
        else:
            raise ValueError(f"Expected output to be (feat, pid) or (feat, pid, camid), got {len(output)} elements")

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats
        q_pids = np.asarray(self.pids)
        q_camids = np.asarray(self.camids) if self.camids else None

        # gallery
        gf = feats
        g_pids = np.asarray(self.pids)
        g_camids = np.asarray(self.camids) if self.camids else None
        
        if self.reranking:
            print("=> Enter reranking")
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print("=> Computing DistMat with euclidean_distance")
            distmat = euclidean_distance(qf, gf)
        
        # 根据 cross_camera 参数决定评估方式
        if self.cross_camera:
            print("=> Cross-camera retrieval enabled: removing same-camera matches")
            if q_camids is None or g_camids is None:
                print("WARNING: cross_camera=True but camera IDs not provided, fallback to standard evaluation")
                cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids)
            else:
                cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, 
                                          max_rank=self.max_rank, cross_camera=True)
        else:
            cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids)

        return cmc, mAP, mINP, distmat
