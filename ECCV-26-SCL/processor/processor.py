import logging
import os
import os.path as osp
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp

from loss.make_loss import get_loss
from loss.triplet_loss import TripletLoss
from loss.nfc import EMAModel, NFCTrainer
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _is_main_process(local_rank):
    return (not dist.is_available()) or (not dist.is_initialized()) or local_rank == 0


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _set_freq_norm_epoch(model, epoch):
    updated = 0
    for module in _unwrap(model).modules():
        if hasattr(module, "set_epoch") and callable(module.set_epoch):
            module.set_epoch(epoch)
            updated += 1
    return updated


def _build_nfc_trainer(cfg):
    if not cfg.CHANGE.METHODS.NFC_TRAINING:
        return None
    return NFCTrainer(
        queue_size=cfg.CHANGE.METHODS.NFC_QUEUE_SIZE,
        k1=cfg.CHANGE.METHODS.NFC_K1,
        k2=cfg.CHANGE.METHODS.NFC_K2,
        temperature=cfg.CHANGE.METHODS.NFC_TEMPERATURE,
        feature_dim=cfg.CHANGE.METHODS.NFC_FEATURE_DIM,
        same_species_weight=cfg.CHANGE.METHODS.NFC_SAME_SPECIES_WEIGHT,
        enable_cross_species=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES,
        cross_species_k1=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_K1,
        cross_species_k2=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_K2,
        cross_species_weight=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_WEIGHT,
        cross_species_margin=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_MARGIN,
        cross_species_metric=cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_METRIC,
    )


def _save_checkpoint(cfg, model, optimizer, scheduler, epoch, local_rank):
    if not _is_main_process(local_rank):
        return
    model_state = _unwrap(model).state_dict()
    checkpoint = {
        "cfg": cfg,
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, osp.join(cfg.OUTPUT_DIR, f"ckpt_{epoch}.pth"))
    if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
        torch.save(model_state, osp.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth"))
    previous = osp.join(cfg.OUTPUT_DIR, f"ckpt_{epoch - 1}.pth")
    if osp.exists(previous):
        os.remove(previous)


def _evaluate(cfg, model, val_loaders, epoch, local_rank, device):
    if not cfg.TEST.EVAL or not val_loaders:
        return

    logger = logging.getLogger("MetaN.Training")
    model.eval()
    for name, val_loader in val_loaders.items():
        evaluator = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_bn = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        evaluator_bn.reset()

        for img, pids, cams, camids, viewids, img_paths in val_loader:
            with torch.no_grad():
                img = img.to(device)
                feat, feat_bn = model(forward_type="main", x=img)
                evaluator.update((feat, pids))
                evaluator_bn.update((feat_bn, pids))

        cmc, mAP, mINP, *_ = evaluator.compute()
        cmc_bn, mAP_bn, mINP_bn, *_ = evaluator_bn.compute()
        if _is_main_process(local_rank):
            logger.info("Validation Results - Epoch %d - %s", epoch, name)
            logger.info("mAP: %.1f%%, mINP: %.1f%%", mAP * 100, mINP * 100)
            for rank in [1, 5, 10]:
                if len(cmc) >= rank:
                    logger.info("Rank-%d: %.1f%%", rank, cmc[rank - 1] * 100)
            logger.info("BN mAP: %.1f%%, BN mINP: %.1f%%", mAP_bn * 100, mINP_bn * 100)

    model.train()


def do_train(
    cfg,
    start_epoch,
    model,
    train_loader,
    val_loaders,
    optimizer,
    scheduler,
    num_domains,
    num_classes,
    num_images,
    num_queries,
    local_rank,
):
    logger = logging.getLogger("MetaN.Training")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if cfg.MODEL.DIST_TRAIN:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    id_loss = F.cross_entropy
    tri_loss = TripletLoss(margin=cfg.SOLVER.MARGIN)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    nfc_trainer = _build_nfc_trainer(cfg)
    ema_model = EMAModel(model, momentum=cfg.CHANGE.METHODS.NFC_EMA_MOMENTUM) if nfc_trainer else None

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    nfc_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS + 1):
        start_time = time.time()
        loss_meter.reset()
        id_loss_meter.reset()
        tri_loss_meter.reset()
        nfc_loss_meter.reset()
        acc_meter.reset()
        _set_freq_norm_epoch(model, epoch)
        scheduler.step(epoch)
        model.train()

        epoch_nfc_stats = {
            "total_samples": 0,
            "total_neighbors": 0,
            "same_species_neighbors": 0,
            "same_id_neighbors": 0,
            "samples_with_neighbors": 0,
            "cross_species_neighbors_count": 0,
        }

        for n_iter, batch in enumerate(train_loader):
            img, pids, cams, views, spes, names, marks = batch
            img = img.to(device, non_blocking=True)
            target_pids = pids.to(device, non_blocking=True)
            target_spes = spes.to(device, non_blocking=True)
            target_marks = marks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                score, feat, extra_info = model(
                    forward_type="main",
                    x=img,
                    label=target_pids,
                    spe_label=target_spes,
                )
                loss, loss_id, loss_tri = get_loss(
                    cfg=cfg,
                    score=score,
                    feat=feat,
                    target_pids=target_pids,
                    target_marks=target_marks,
                    id_loss=id_loss,
                    tri_loss=tri_loss,
                )

                loss_nfc = torch.zeros((), device=device)
                if nfc_trainer is not None and ema_model is not None and epoch > cfg.CHANGE.METHODS.NFC_WARMUP_EPOCHS:
                    ema_model.apply_shadow()
                    with torch.no_grad():
                        _, teacher_feat, _ = model(
                            forward_type="main",
                            x=img,
                            label=target_pids,
                            spe_label=target_spes,
                        )
                    ema_model.restore()
                    loss_nfc, batch_stats = nfc_trainer.compute_nfc_loss(
                        feat,
                        teacher_feat,
                        target_pids,
                        device,
                        spes=target_spes,
                    )
                    nfc_trainer.update_queue(teacher_feat, target_pids, spes=target_spes)
                    for key in epoch_nfc_stats:
                        epoch_nfc_stats[key] += batch_stats.get(key, 0)
                    loss = loss + cfg.CHANGE.METHODS.NFC_LOSS_WEIGHT * loss_nfc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_model.update()

            batch_size = img.size(0)
            acc = (score.max(1)[1] == target_pids).float().mean()
            loss_meter.update(loss.detach().item(), batch_size)
            id_loss_meter.update(loss_id.detach().item(), batch_size)
            tri_loss_meter.update(loss_tri.detach().item(), batch_size)
            nfc_loss_meter.update(loss_nfc.detach().item(), batch_size)
            acc_meter.update(acc.item(), 1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if _is_main_process(local_rank) and (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                warmup = nfc_trainer is not None and epoch <= cfg.CHANGE.METHODS.NFC_WARMUP_EPOCHS
                logger.info(
                    "Epoch[%d/%d] Iteration[%d/%d] Loss: %.3f, ID: %.3f, Triplet: %.3f, NFC: %.3f, Acc: %.3f, LR: %.2e%s",
                    epoch,
                    cfg.SOLVER.MAX_EPOCHS,
                    n_iter + 1,
                    len(train_loader),
                    loss_meter.avg,
                    id_loss_meter.avg,
                    tri_loss_meter.avg,
                    nfc_loss_meter.avg,
                    acc_meter.avg,
                    scheduler._get_lr(epoch)[0],
                    " (NFC warmup)" if warmup else "",
                )

        if _is_main_process(local_rank):
            time_per_batch = (time.time() - start_time) / max(len(train_loader), 1)
            logger.info(
                "Epoch %d done. Time per batch: %.3fs, speed: %.1f samples/s",
                epoch,
                time_per_batch,
                train_loader.batch_size / time_per_batch if hasattr(train_loader, "batch_size") and train_loader.batch_size else 0.0,
            )
            if nfc_trainer is not None and epoch > cfg.CHANGE.METHODS.NFC_WARMUP_EPOCHS:
                total_neighbors = epoch_nfc_stats["total_neighbors"]
                if total_neighbors > 0:
                    same_species = epoch_nfc_stats["same_species_neighbors"] / total_neighbors
                    same_id = epoch_nfc_stats["same_id_neighbors"] / total_neighbors
                    logger.info(
                        "NFC stats: same_species=%.2f%%, same_id=%.2f%%, cross_neighbors=%d",
                        same_species * 100,
                        same_id * 100,
                        epoch_nfc_stats["cross_species_neighbors_count"],
                    )

        _save_checkpoint(cfg, model, optimizer, scheduler, epoch, local_rank)
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            _evaluate(cfg, model, val_loaders, epoch, local_rank, device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if _is_main_process(local_rank):
        last_ckpt = osp.join(cfg.OUTPUT_DIR, f"ckpt_{cfg.SOLVER.MAX_EPOCHS}.pth")
        if osp.exists(last_ckpt):
            os.remove(last_ckpt)


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("MetaN.Test")
    evaluator = R1_mAP_eval(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.to(device)
    model.eval()

    for img, pids, cams, camids, viewids, img_paths in val_loader:
        with torch.no_grad():
            img = img.to(device)
            feat, _ = model(forward_type="main", x=img)
            evaluator.update((feat, pids))

    cmc, mAP, mINP, *_ = evaluator.compute()
    logger.info("mAP: %.1f%%, mINP: %.1f%%", mAP * 100, mINP * 100)
    for rank in [1, 5, 10]:
        logger.info("CMC Rank-%d: %.1f%%", rank, cmc[rank - 1] * 100)
