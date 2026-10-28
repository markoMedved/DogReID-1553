def get_loss(cfg, score, feat, target_pids, target_marks, id_loss, tri_loss):
    loss_id = id_loss(score, target_pids)
    loss_tri = tri_loss(feat, target_pids, marks=target_marks)[0]
    loss = (
        loss_id * cfg.MODEL.ID_LOSS_WEIGHT
        + loss_tri * cfg.MODEL.TRIPLET_LOSS_WEIGHT
    )
    return loss, loss_id, loss_tri
