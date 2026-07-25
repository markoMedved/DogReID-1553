# Re-ID methodology

This document describes the re-identification pipeline, the methods implemented
in it, how to run them, and what to record from each run.

- [1. Pipeline](#1-pipeline)
- [2. Methods](#2-methods)
- [3. Integration](#3-integration)
- [4. Running experiments](#4-running-experiments)
- [5. Reporting](#5-reporting)
- [6. Tests](#6-tests)

---

## 1. Pipeline

```
splits.csv ─┐
            ├─> data/dataset.py        DogVideoDataset
bounding_   │     · samples clip_len frames per video
boxes.csv ──┘     · crops to the dog (ground-truth box on the first frame,
                    YOLO detection otherwise)
                  · data/reid_transforms.py
                        resize, horizontal flip, pad and crop, normalize,
                        random erasing, sampled once per clip
                        |
            data/dataloader.py         PK sampler: P identities x K clips
                        |  (B, T, C, H, W)
            models/reid_model.py       VideoReID
              |- backbone adapter      DINOv2 or timm
              |    · bot        pooled class-token feature per frame
              |    · transreid  penultimate tokens -> JPM -> 1 global + k local
              |- TemporalPool          attention, mean or max, per branch
              |- BNNeckHead per branch
                   · triplet feature   before the bottleneck
                   · identity logits   after the bottleneck
                   · retrieval feature after the bottleneck, L2-normalized
                        |
            engine/trainer.py          triplet loss + identity loss
              · AdamW, gradient accumulation
              · solver/warmup.py       linear warmup, then step decay
                        |
            evaluation/make_csv.py     cosine distance matrix per protocol
            evaluation/*.ipynb         mAP, Rank-k, DIR@FAR, bootstrap CIs
```

Freezing is applied once after the model is built, by `models/freezing.py`.

### Model outputs

| | Training | Inference |
|---|---|---|
| `VideoReID.forward` returns | `(embeddings, logits)` | `embeddings` |
| `embeddings` | pre-bottleneck global feature, L2-normalized | — |
| retrieval feature | — | `bot`: 768-D. `transreid`: concatenation of the global and k local features, L2-normalized |

---

## 2. Methods

### 2.1 BoT baseline (`reid_method="bot"`)

Luo, Jiang, Gu, Liao, Lai, Gu. *Bag of Tricks and a Strong Baseline for Deep
Person Re-Identification.* CVPR Workshops 2019. Extended as *A Strong Baseline
and Batch Normalization Neck for Deep Person Re-Identification*, IEEE
Transactions on Multimedia 22(10), 2020. arXiv:1903.07071.
Reference implementation: https://github.com/michuanhaohao/reid-strong-baseline

| Component | Location |
|---|---|
| BNNeck, triplet before and identity after the bottleneck (Sec. 3.2) | `models/reid_heads.py` |
| Identity loss with label smoothing 0.1 | `engine/trainer.py` |
| Triplet loss with hard mining | `engine/trainer.py` |
| Warmup learning rate (Sec. 3.2) | `solver/warmup.py` |
| Random erasing (Sec. 3.2) | `data/reid_transforms.py` |
| Horizontal flip, pad and random crop | `data/reid_transforms.py` |

Two components of the original recipe are not used. Last-stride reduction is a
ResNet modification with no ViT equivalent, and center loss is optional in the
original work and is not applied here.

### 2.2 TransReID (`reid_method="transreid"`)

He, Luo, Wang, Wang, Li, Jiang. *TransReID: Transformer-based Object
Re-Identification.* ICCV 2021. arXiv:2102.04378.
Reference implementation: https://github.com/damo-cv/TransReID

The Jigsaw Patch Module (Sec. 3.3) is implemented in `models/reid_heads.py`:

- The penultimate token sequence feeds a global and a local branch.
- The global branch uses the backbone's final block; the local branch uses an
  independent copy, so the branches do not share weights.
- Patch tokens are shift-and-shuffled (shift 5, 2 interleaved groups) and split
  into k = 4 groups, each re-prefixed with the class token.
- Each of the k + 1 branches has its own BNNeck and classifier.
- At inference the k + 1 features are concatenated.

Side Information Embedding (Sec. 3.2) is not used. DogReID-1553 provides no
camera or viewpoint labels, and the only available side information is the scene
group, which the scene-disjoint evaluation protocol is designed to exclude as an
identity cue.

TransReID is an image-level method. Frame features are aggregated with the same
temporal pooling as the other benchmark models, applied per branch.

### 2.3 Input resolution

The dog bounding boxes in `bounding_boxes.csv` have a geometric-mean
width:height ratio of 0.81 across all 7,463 annotated frames, so the default
input is non-square. Use 256x192 for patch-16 backbones and 252x182 for DINOv2,
whose patch size is 14.

### 2.4 Backbones and other references

- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.*
  TMLR 2024. arXiv:2304.07193
- Zhong, Zheng, Kang, Li, Yang. *Random Erasing Data Augmentation.* AAAI 2020.
  arXiv:1708.04896
- Hermans, Beyer, Leibe. *In Defense of the Triplet Loss for Person
  Re-Identification.* arXiv:1703.07737, 2017

Additional backbones are loaded from `timm` and provide pooled features only.

---

## 3. Integration

Five existing files are involved. `engine/trainer.py` is already updated; the rest need edits.

### 3.1 `configs/config.py`

Add the following fields:

```python
backbone      = "dinov2"      # dinov2, vit, swin, convnext
reid_method   = "bot"         # bot, transreid
img_size      = (252, 182)    # 256x192 for patch-16 backbones
full_finetune = True
jpm_parts     = 4

warmup_epochs = 10
lr_milestones = (40, 70)
lr_gamma      = 0.1
aug_pad       = 10
re_prob       = 0.5
```

Extend `run_name` so that `reid_method` and `backbone` appear in output paths:

```python
self.run_name = (f"{self.backbone}_{self.reid_method}_{self.world}"
                 f"_{self.pooling_type}_finetune_{self.full_finetune}")
```

### 3.2 `models/model_factory.py`

```python
from .reid_model import VideoReID

def build_model(cfg):
    if getattr(cfg, "reid_method", None) in ("bot", "transreid"):
        return VideoReID(cfg)
    # existing branches follow
```

### 3.3 `engine/trainer.py`

Already updated. `Trainer` now takes an optional `scheduler`, steps it once per
epoch, unpacks models that return `(embeddings, logits)`, and adds the identity
loss when logits are present:

```python
Trainer(model, train_loader, query_loader, gallery_loader,
        optimizer, cfg, loss_fn, miner, scheduler=scheduler)
```

Models returning a single tensor keep their previous behaviour, so the existing
builders are unaffected. The identity loss uses cross entropy with label
smoothing 0.1, weighted by `cfg.id_loss_weight` (default 1.0).

### 3.4 `train.py`

Set `num_classes` from the training split, apply the freezing policy, build the
scheduler and pass it to the trainer. Replace the inline freezing block with:

```python
from models.freezing import apply_freezing, trainable_report
from solver.warmup import build_scheduler

train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

# train_loader.dataset is a Subset, so id_map lives one level deeper
cfg.num_classes = len(train_loader.dataset.dataset.id_map)
print(f"[cfg] num_classes = {cfg.num_classes}")

model = build_model(cfg).to(cfg.device)
model = apply_freezing(model, cfg)
print(trainable_report(model))

scheduler = build_scheduler(optimizer, cfg)
```

and pass `scheduler=scheduler` in the `Trainer(...)` call.

### 3.5 `data/dataloader.py`

Use the re-ID transform pipeline:

```python
from data.reid_transforms import build_video_transforms

train_tf = build_video_transforms(cfg, is_train=True)
eval_tf  = build_video_transforms(cfg, is_train=False)
```

### 3.6 `evaluation/make_csv.py`

The script builds a model with `MODEL_CLASS()` and no arguments. Add a branch
that constructs `VideoReID` from the configuration instead, and extend the
`--model_name` choices:

```python
choices=["dinov2", "swin", "vit", "bot", "transreid"]
```

```python
if MODEL_NAME in ("bot", "transreid"):
    from models.reid_model import VideoReID
    cfg.reid_method = MODEL_NAME
    cfg.num_classes = 776          # must match the training run
    model = VideoReID(cfg)
else:
    model = MODEL_CLASS()
```

`cfg.num_classes` must be the value used during training even though the
classifiers are unused at inference. Building the model with `num_classes=0`
omits them, and `load_state_dict` then fails on the unexpected
`heads.*.classifier.weight` entries in the checkpoint.

`cfg.backbone`, `cfg.img_size`, `cfg.pooling_type` and `cfg.jpm_parts` must also
match the training run, or the checkpoint will not load.

---

## 4. Running experiments

### 4.1 Prerequisites

```bash
pip install -r requirements.txt
python tests/test_reid_smoke.py
```

DINOv2 weights are fetched through `torch.hub` on first use. On a cluster without
outbound network access from compute nodes, warm the cache on a login node first:

```bash
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')"
```

### 4.2 Training

Each configuration is trained separately for the closed-world and open-world
splits, because the splits differ. Checkpoints are written to
`trained_models/{model_name}_{world}/model.pth`.

```bash
# BoT baseline
python train.py            # cfg.reid_method="bot",       cfg.world="closed"
python train.py            # cfg.reid_method="bot",       cfg.world="open"

# TransReID
python train.py            # cfg.reid_method="transreid",  cfg.world="closed"
python train.py            # cfg.reid_method="transreid",  cfg.world="open"
```

### 4.3 Evaluation

Four invocations per trained configuration produce the distance matrices:

```bash
python evaluation/make_csv.py --model_name bot --world_type closed
python evaluation/make_csv.py --model_name bot --world_type closed --use_images
python evaluation/make_csv.py --model_name bot --world_type open
python evaluation/make_csv.py --model_name bot --world_type open   --use_images
```

Then run `evaluation/closed_set_plots.ipynb` and `evaluation/open_set_plots.ipynb`
to compute metrics and bootstrap confidence intervals.

### 4.4 Experiment matrix

Main results, four trainings:

| Run | backbone | reid_method | full_finetune | pooling | Purpose |
|---|---|---|---|---|---|
| A | dinov2 | bot | True | attention | Strong baseline, main result |
| B | dinov2 | transreid | True | attention | Method result, main result |

Ablations, each varying one field from run A:

| Run | Field changed | Purpose |
|---|---|---|
| C | `full_finetune = False` | Frozen versus fully fine-tuned backbone |
| D | `cfg.num_classes = 0` after the assignment in `train.py` | Triplet loss only, without the identity loss |
| E | `pooling_type = "mean"` | Temporal aggregation |
| F | `pooling_type = "max"` | Temporal aggregation |
| G | `img_size = (224, 224)` | Square versus non-square input |

Runs C to G only need the closed-world split and the video-to-video protocol,
which is one training and one evaluation each.

---

## 5. Reporting

### 5.1 Per-run record

Record the following for every run, from the training log and configuration
dump:

| Field | Source |
|---|---|
| backbone, reid_method, pooling_type, full_finetune | `cfg.run_name` |
| Trainable parameter count and percentage | `trainable_report()`, first line of the log |
| Input size | `cfg.img_size` |
| Epochs, effective batch size, learning rate, margin | `cfg` dump |
| Identity count | `cfg.num_classes` |
| Retrieval feature dimension | 768 for `bot`, 3840 for `transreid` with k=4 |
| Wall-clock time per epoch | training log |

### 5.2 Closed-world table

Report mAP, Rank-1 and Rank-5 with 95 % bootstrap confidence intervals, for each
of the three query protocols, matching the existing results tables:

```
Video-to-Video
Image-to-Video
Image-to-Image
```

### 5.3 Open-world table

Report DIR at FAR = 0.01, 0.05 and 0.10 with 95 % bootstrap confidence
intervals, for Video-to-Video and Image-to-Video.

### 5.4 Ablation tables

| Ablation | Rows to report | Metric |
|---|---|---|
| Fine-tuning (A vs C) | frozen, full | mAP, Rank-1, Rank-5, plus trainable % |
| Loss (A vs D) | triplet, triplet + identity | mAP, Rank-1, Rank-5 |
| Temporal pooling (A vs E vs F) | attention, mean, max | mAP, Rank-1, Rank-5 |
| Input resolution (A vs G) | 252x182, 224x224 | mAP, Rank-1, Rank-5 |

### 5.5 Method configuration to report

State these alongside the results so the runs can be reproduced:

| Setting | Value |
|---|---|
| Identities, closed-world training split | 776 |
| Training videos, closed / open | 3,788 / 3,708 |
| PK sampling | P identities x K clips; 106 of 776 training identities have fewer than 4 videos |
| Input size | 252x182 (DINOv2), 256x192 (patch-16 backbones) |
| Augmentation | flip p=0.5, pad 10 and random crop, random erasing p=0.5 (sl=0.02, sh=0.4, r1=0.3) |
| Loss | hard-mined triplet + identity loss with label smoothing 0.1 |
| Optimizer | AdamW with gradient accumulation |
| Schedule | linear warmup for 10 epochs, then x0.1 at epochs 40 and 70 |
| JPM | k=4 groups, shift 5, 2 shuffle groups |
| SIE | not used, see Sec. 2.2 |

Comparisons across runs are only valid at a fixed input size. If run G is
included, report the input size in every row.

---

## 6. Tests

```bash
python tests/test_reid_smoke.py
REID_DEVICE=mps python tests/test_reid_smoke.py
```

The tests cover the jigsaw permutation, BNNeck ordering, forward and backward
passes for both methods, the freezing policy, the transform pipeline and the
learning rate schedule. They use synthetic tensors and stub backbones, so they
verify shapes and behaviour only, not accuracy.
