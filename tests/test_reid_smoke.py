"""Shape and behaviour tests for the re-ID model, heads and transforms.

Runs on synthetic tensors only: no dataset, no pretrained weights, no accuracy
measurement. Backbones are replaced by a small stub transformer so the tests run
without network access.

Usage:
    python tests/test_reid_smoke.py                  # cuda, else mps, else cpu
    REID_DEVICE=mps python tests/test_reid_smoke.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.reid_transforms import build_transforms, build_video_transforms  # noqa: E402
from models import reid_model  # noqa: E402
from models.freezing import apply_freezing, trainable_report  # noqa: E402
from models.reid_heads import JPM, BNNeckHead, shuffle_patches  # noqa: E402
from models.reid_model import TemporalPool, VideoReID  # noqa: E402
from solver.warmup import WarmupMultiStepLR  # noqa: E402


def pick_device():
    forced = os.environ.get("REID_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

D = 64
B, T, H, W = 2, 4, 28, 28
PATCH = 4
PATCHES = (H // PATCH) * (W // PATCH)
NUM_CLASSES = 7


# --- Stub Backbones ---

class StubBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class StubViT(nn.Module):
    """Minimal transformer exposing the same surface as DINOv2's backbone."""

    def __init__(self, dim=D, depth=3, patches=PATCHES):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, patches + 1, dim))
        self.blocks = nn.ModuleList([StubBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def prepare_tokens_with_masks(self, x, masks=None):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        return t + self.pos[:, : t.shape[1]]

    def forward(self, x):
        t = self.prepare_tokens_with_masks(x)
        for block in self.blocks:
            t = block(t)
        return self.norm(t)[:, 0]


class StubAdapter(reid_model.DINOv2Adapter):
    """StubViT behind the real DINOv2 adapter, so penultimate(), pooled() and
    the no_grad split under test are production code rather than a copy."""

    def __init__(self):
        nn.Module.__init__(self)
        self.net = StubViT()
        self.dim = D


class PooledOnlyAdapter(nn.Module):
    """Adapter without a token sequence, as for Swin or ConvNeXt."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, D))
        self.dim = D

    def penultimate(self, x, n_grad=None):
        raise NotImplementedError

    def pooled(self, x, n_grad=None):
        return self.net(x)


# --- Fixtures ---

class Cfg:
    backbone = "stub"
    reid_method = "bot"
    pooling_type = "attention"
    num_classes = NUM_CLASSES
    chunk_size = 4
    img_size = (H, W)
    full_finetune = False
    jpm_parts = 4


def make_model(method="bot", pooling="attention", adapter=StubAdapter, full_ft=False):
    cfg = Cfg()
    cfg.reid_method = method
    cfg.pooling_type = pooling
    cfg.full_finetune = full_ft

    original = reid_model.build_backbone
    reid_model.build_backbone = lambda _cfg: adapter()
    try:
        return VideoReID(cfg).to(DEVICE), cfg
    finally:
        reid_model.build_backbone = original


def mock_clip():
    return torch.zeros(B, T, 3, H, W, device=DEVICE)


def mock_labels():
    return torch.arange(B, device=DEVICE) % NUM_CLASSES


# --- Tests ---

def test_shuffle_is_permutation():
    x = torch.arange(PATCHES, dtype=torch.float32).view(1, PATCHES, 1)
    out = shuffle_patches(x, shift=5, groups=2)

    assert out.shape == x.shape, f"shape changed: {out.shape}"
    assert sorted(out.flatten().tolist()) == sorted(x.flatten().tolist()), "tokens lost or duplicated"
    assert not torch.equal(out, x), "shuffle had no effect"


def test_bnneck_ordering():
    head = BNNeckHead(D, NUM_CLASSES).to(DEVICE).train()
    feat = torch.randn(4, D, device=DEVICE)

    triplet_feat, retrieval_feat, logits = head(feat)

    assert torch.equal(triplet_feat, feat), "triplet feature must be taken before the bottleneck"
    assert logits.shape == (4, NUM_CLASSES)
    assert torch.allclose(retrieval_feat.norm(dim=-1), torch.ones(4, device=DEVICE), atol=1e-5)

    head.eval()
    assert head(feat)[2] is None, "logits should not be produced at inference"


def test_temporal_pool_modes():
    x = torch.randn(B, T, D, device=DEVICE)

    for mode in ("attention", "mean", "max"):
        assert TemporalPool(D, mode).to(DEVICE)(x).shape == (B, D), mode

    assert torch.allclose(TemporalPool(D, "mean").to(DEVICE)(x), x.mean(1), atol=1e-6)
    assert torch.allclose(TemporalPool(D, "max").to(DEVICE)(x), x.max(1).values, atol=1e-6)

    try:
        TemporalPool(D, "median")
    except ValueError:
        return
    raise AssertionError("an unknown pooling mode should raise ValueError")


def test_bot_forward_backward():
    model, _ = make_model("bot")
    model.train()

    embeddings, logits = model(mock_clip())
    assert embeddings.shape == (B, D), embeddings.shape
    assert logits.shape == (B, NUM_CLASSES), logits.shape

    loss = nn.functional.cross_entropy(logits, mock_labels()) + embeddings.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    model.eval()
    with torch.no_grad():
        assert model(mock_clip()).shape == (B, D)


def test_transreid_forward_backward():
    model, cfg = make_model("transreid")
    model.train()

    embeddings, logits = model(mock_clip())
    assert embeddings.shape == (B, D), embeddings.shape
    assert logits.shape == (B, NUM_CLASSES), logits.shape

    loss = nn.functional.cross_entropy(logits, mock_labels()) + embeddings.pow(2).mean()
    loss.backward()

    local_block = model.jpm.local_block
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in local_block.parameters()), \
        "the JPM local branch received no gradient"

    model.eval()
    with torch.no_grad():
        expected = (B, D * (cfg.jpm_parts + 1))
        assert model(mock_clip()).shape == expected, expected


def test_transreid_requires_token_backbone():
    try:
        make_model("transreid", adapter=PooledOnlyAdapter)
    except ValueError as exc:
        assert "transreid" in str(exc).lower()
        return
    raise AssertionError("transreid should reject a backbone without a token sequence")


def test_full_finetune_unfreezes_everything():
    model, cfg = make_model("transreid", full_ft=True)
    apply_freezing(model, cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == total, f"{trainable} of {total} trainable"


def test_partial_finetune_keeps_heads_trainable():
    model, cfg = make_model("transreid", full_ft=False)
    apply_freezing(model, cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < total, f"{trainable} of {total} trainable"

    for i, head in enumerate(model.heads):
        assert all(p.requires_grad for p in head.classifier.parameters()), f"head {i} classifier"
        assert all(p.requires_grad for p in head.bottleneck.parameters()), f"head {i} bottleneck"
    for i, pool in enumerate(model.pools):
        assert all(p.requires_grad for p in pool.parameters()), f"pool {i}"
    assert all(p.requires_grad for p in model.jpm.local_block.parameters())

    assert not any(p.requires_grad for p in model.backbone.net.blocks[0].parameters()), \
        "early backbone blocks should stay frozen"

    report = trainable_report(model)
    assert "trainable" in report and "%" in report, report


def test_unfreeze_blocks_count():
    """cfg.unfreeze_blocks controls how many trailing backbone blocks train.
    n=0 must unfreeze none of them, not all of them."""
    def backbone_trainable(n):
        model, cfg = make_model("bot", full_ft=False)
        cfg.unfreeze_blocks = n
        apply_freezing(model, cfg)
        blocks = model.backbone.net.blocks
        return [any(p.requires_grad for p in b.parameters()) for b in blocks]

    assert backbone_trainable(0) == [False, False, False], backbone_trainable(0)
    assert backbone_trainable(1) == [False, False, True], backbone_trainable(1)
    assert backbone_trainable(2) == [False, True, True], backbone_trainable(2)

    # Heads stay trainable regardless of the backbone setting
    model, cfg = make_model("bot", full_ft=False)
    cfg.unfreeze_blocks = 0
    apply_freezing(model, cfg)
    assert all(p.requires_grad for p in model.heads[0].bottleneck.parameters())


def test_frozen_prefix_saves_memory_without_changing_output():
    """Running the frozen leading blocks under no_grad must not change the
    result, and must leave those blocks out of the autograd graph."""
    torch.manual_seed(0)
    model, cfg = make_model("bot", full_ft=False)
    cfg.unfreeze_blocks = 2
    apply_freezing(model, cfg)
    model.train()

    x = torch.randn(B, T, 3, H, W, device=DEVICE)

    # Same arithmetic with and without the no_grad split
    model.n_grad = None
    full_graph = model(x)[0]
    model.n_grad = 2
    split_graph = model(x)[0]
    assert torch.allclose(full_graph, split_graph, atol=1e-5), \
        (full_graph - split_graph).abs().max().item()

    # Only the trailing blocks should receive gradient
    model.zero_grad()
    split_graph.pow(2).mean().backward()
    blocks = model.backbone.net.blocks
    got_grad = [any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters())
                for b in blocks]
    assert got_grad[-1], "last block should receive gradient"
    assert not got_grad[0], "frozen leading block should be outside the graph"


def test_parameter_name_prefixes():
    """train.py splits learning rates by parameter name: 'backbone*' and 'jpm*'
    are pretrained and get the reduced rate, everything else is randomly
    initialized and gets the full rate. If a module is renamed, that split
    silently mis-assigns learning rates, so the prefixes are pinned here."""
    expected_pretrained = {"backbone", "jpm"}
    expected_random = {"heads", "pools"}

    for method in ("bot", "transreid"):
        model, cfg = make_model(method, full_ft=False)
        apply_freezing(model, cfg)
        prefixes = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
        unknown = prefixes - expected_pretrained - expected_random
        assert not unknown, f"{method}: unrecognised parameter prefixes {unknown}"

        # Coverage: the two groups must partition the trainable parameters
        pre = [p for n, p in model.named_parameters()
               if p.requires_grad and (n.startswith("backbone") or n.startswith("jpm."))]
        rnd = [p for n, p in model.named_parameters()
               if p.requires_grad and not (n.startswith("backbone") or n.startswith("jpm."))]
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert sum(p.numel() for p in pre) + sum(p.numel() for p in rnd) == total, method

    # TransReID's JPM copy must be counted as pretrained, not as a fresh head
    model, cfg = make_model("transreid", full_ft=False)
    apply_freezing(model, cfg)
    jpm_names = [n for n, p in model.named_parameters()
                 if p.requires_grad and n.startswith("jpm.")]
    assert jpm_names, "JPM parameters missing from the trainable set"


def test_osnet_backbone():
    """OSNet from Torchreid: pooled features only, rejects transreid, and the
    freezing policy recognises its conv-stage layout. Skipped when torchreid is
    not installed."""
    try:
        import importlib
        importlib.import_module("torchreid.reid.models.osnet")
    except ImportError:
        print("      (skipped: torchreid not installed)", end="")
        return

    from configs.config import Config
    cfg = Config()
    cfg.backbone = "osnet"
    cfg.reid_method = "bot"
    cfg.num_classes = NUM_CLASSES
    cfg.osnet_pretrained = False       # no download in tests
    cfg.chunk_size = 4

    model = VideoReID(cfg).to(DEVICE)
    assert model.embed_dim == 512, model.embed_dim

    apply_freezing(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total, f"{trainable}/{total}"

    x = torch.zeros(2, 4, 3, 224, 224, device=DEVICE)
    model.train()
    emb, logits = model(x)
    assert emb.shape == (2, 512) and logits.shape == (2, NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == (2, 512)

    cfg.reid_method = "transreid"
    try:
        VideoReID(cfg)
    except ValueError:
        return
    raise AssertionError("transreid should be rejected on a CNN backbone")


def test_megadescriptor_backbones():
    """MegaDescriptor (wildlife-pretrained Swin fetched from the HuggingFace hub
    via timm), both variants: the standalone 'without BoT' builder and the
    'with BoT' VideoReID backbone. timm.create_model is stubbed with a tiny
    Swin-shaped module so the test needs no network access or pretrained
    weights."""
    import types

    class FakeSwin(nn.Module):
        """Minimal stand-in exposing timm SwinTransformer's surface: a pooled
        forward, .num_features, and the .layers/.norm layout the freezer walks."""

        def __init__(self, dim=D):
            super().__init__()
            self.num_features = dim
            self.proj = nn.Linear(3, dim)
            # Two stages plus a final norm, matching what _unfreeze_last_blocks
            # looks for on a Swin backbone.
            self.layers = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            # (N, 3, H, W) -> global-pool over space -> (N, dim)
            return self.proj(x.mean(dim=(2, 3)))

    fake_timm = types.ModuleType("timm")
    fake_timm.create_model = lambda name, pretrained=True, num_classes=0: FakeSwin(D)

    saved = sys.modules.get("timm")
    sys.modules["timm"] = fake_timm
    try:
        # --- without BoT: standalone builder ---
        from models.megadescriptor_builder import MegaDescriptor
        mega = MegaDescriptor(variant="hf-hub:BVRA/MegaDescriptor-L-224", chunk_size=4).to(DEVICE)
        assert mega.dim == D, mega.dim
        mega.eval()
        with torch.no_grad():
            out = mega(mock_clip())
        assert out.shape == (B, D), out.shape
        # Output must be L2-normalized for cosine retrieval
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, device=DEVICE), atol=1e-4)

        # --- with BoT: VideoReID backbone ---
        from configs.config import Config
        cfg = Config()
        cfg.backbone = "megadescriptor"
        cfg.reid_method = "bot"
        cfg.num_classes = NUM_CLASSES
        cfg.chunk_size = 4

        model = VideoReID(cfg).to(DEVICE)
        assert model.embed_dim == D, model.embed_dim

        apply_freezing(model, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert 0 < trainable < total, f"{trainable}/{total}"

        model.train()
        emb, logits = model(mock_clip())
        assert emb.shape == (B, D) and logits.shape == (B, NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            assert model(mock_clip()).shape == (B, D)

        # Swin exposes no flat token sequence, so transreid must be rejected
        cfg.reid_method = "transreid"
        try:
            VideoReID(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("transreid should be rejected on a Swin backbone")
    finally:
        if saved is not None:
            sys.modules["timm"] = saved
        else:
            del sys.modules["timm"]


def test_miewid_efficientnet_freezing():
    """MiewID wraps a timm EfficientNetV2 under `.backbone` inside its HF module
    (MiewIdNet). The freezer must descend through that wrapper and unfreeze the
    last conv stage + final projection rather than raise 'unrecognized layout'.
    transformers is stubbed so the test needs neither the package nor network."""
    import types

    class FakeEffNet(nn.Module):
        """timm EfficientNet surface: conv_stem -> blocks -> conv_head -> bn2 -> pool."""

        def __init__(self, dim=D):
            super().__init__()
            self.conv_stem = nn.Conv2d(3, dim, 3, padding=1)
            self.blocks = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 1))
            self.conv_head = nn.Conv2d(dim, dim, 1)
            self.bn2 = nn.BatchNorm2d(dim)
            self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        def forward(self, x):
            x = self.blocks(self.conv_stem(x))
            x = self.bn2(self.conv_head(x))
            return self.global_pool(x)

    class FakeMiewIdNet(nn.Module):
        """Mimics HF MiewIdNet: timm backbone under `.backbone` plus a BN neck."""

        def __init__(self, dim=D):
            super().__init__()
            self.backbone = FakeEffNet(dim)
            self.bn = nn.BatchNorm1d(dim)

        def forward(self, x):
            return self.bn(self.backbone(x))

    def fake_from_pretrained(*args, **kwargs):
        # transformers.from_pretrained returns eval mode; mirror it so the
        # builder's batch-1 dimension probe does not trip BatchNorm1d.
        model = FakeMiewIdNet(D)
        model.eval()
        return model

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModel = types.SimpleNamespace(from_pretrained=fake_from_pretrained)

    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_tf
    try:
        from models.miewid_builder import MiewIDReID
        model = MiewIDReID(chunk_size=4).to(DEVICE)
        assert model.dim == D, model.dim

        cfg = types.SimpleNamespace(full_finetune=False, unfreeze_blocks=2)
        # Must not raise "Unrecognized backbone layout"
        apply_freezing(model, cfg)

        effnet = model.backbone.backbone
        assert any(p.requires_grad for p in effnet.blocks[-1].parameters()), "last stage frozen"
        assert any(p.requires_grad for p in effnet.conv_head.parameters()), "conv_head frozen"
        assert not any(p.requires_grad for p in effnet.conv_stem.parameters()), \
            "stem should stay frozen"
        assert all(p.requires_grad for p in model.temporal_pool.parameters()), "temporal pool frozen"
        assert model.bn.weight.requires_grad, "neck bn frozen"

        model.eval()
        with torch.no_grad():
            out = model(mock_clip())
        assert out.shape == (B, D), out.shape
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, device=DEVICE), atol=1e-4)
    finally:
        if saved is not None:
            sys.modules["transformers"] = saved
        else:
            del sys.modules["transformers"]


def test_checkpoint_classifier_keys():
    """evaluation/make_csv.py recovers num_classes from the checkpoint by
    matching 'heads.*.classifier.weight'. A backbone that ships its own
    classifier would otherwise be picked up instead, so the state_dict must
    contain no other classifier weights."""
    model, _ = make_model("transreid", full_ft=False)
    state = model.state_dict()

    all_cls = [k for k in state if k.endswith("classifier.weight")]
    head_cls = [k for k in all_cls if k.startswith("heads.")]
    assert all_cls == head_cls, f"non-head classifier weights in checkpoint: {set(all_cls) - set(head_cls)}"
    assert head_cls, "no classifier weights saved"

    detected = state[head_cls[0]].shape[0]
    assert detected == NUM_CLASSES, f"detected {detected}, expected {NUM_CLASSES}"


def test_unknown_backbone_layout_raises():
    class UnknownAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 4)
            self.dim = D

        def penultimate(self, x):
            raise NotImplementedError

        def pooled(self, x):
            return torch.zeros(x.shape[0], D, device=x.device)

    model, cfg = make_model("bot", adapter=UnknownAdapter)
    try:
        apply_freezing(model, cfg)
    except RuntimeError:
        return
    raise AssertionError("an unrecognized backbone layout should raise")


def test_dinov2_adapter_token_path():
    original = torch.hub.load
    torch.hub.load = lambda *args, **kwargs: StubViT()
    try:
        adapter = reid_model.DINOv2Adapter("vitb14_reg").to(DEVICE)
        adapter.dim = D

        x = torch.zeros(2, 3, H, W, device=DEVICE)
        tokens = adapter.penultimate(x)
        assert tokens.shape == (2, PATCHES + 1, D), tokens.shape
        assert adapter.pooled(x).shape == (2, D)

        feats = JPM(adapter.last_block, adapter.final_norm, k=4).to(DEVICE)(tokens)
        assert len(feats) == 5
        assert all(f.shape == (2, D) for f in feats)
    finally:
        torch.hub.load = original


def test_transforms():
    from PIL import Image

    cfg = Cfg()
    cfg.img_size = (64, 48)
    img = Image.new("RGB", (200, 300), color=(120, 120, 120))

    assert build_transforms(cfg, is_train=False)(img).shape == (3, 64, 48)
    assert build_transforms(cfg, is_train=True)(img).shape == (3, 64, 48)

    clip = build_video_transforms(cfg, is_train=True)([img] * 5)
    assert clip.shape == (5, 3, 64, 48), clip.shape
    assert torch.allclose(clip[0], clip[-1], atol=1e-6), \
        "augmentation must be identical across frames of one clip"


def test_run_name_round_trip():
    """The checkpoint directory written by training must be the one evaluation
    looks for, otherwise a finished run is unreadable."""
    from configs.config import Config

    cfg = Config()
    training_name = cfg.run_name

    # Evaluation rebuilds the name from its own flags
    eval_name = Config.compose_run_name(
        backbone=Config.backbone,
        reid_method=cfg.reid_method,
        world=cfg.world,
        pooling_type=cfg.pooling_type,
        full_finetune=cfg.full_finetune,
    )
    assert training_name == eval_name, f"{training_name!r} != {eval_name!r}"

    # Overrides must propagate into the name, not leave a stale one behind
    cfg.reid_method = "transreid"
    cfg.world = "open"
    cfg.batch_size, cfg.k = 12, 3
    cfg.refresh_run_name(make_dir=False)
    assert cfg.run_name == "dinov2_transreid_open_attention_finetune_False", cfg.run_name
    assert cfg.num_ids == 4, cfg.num_ids
    assert cfg.output_dir.name == cfg.run_name


def test_schedule_fits_epochs():
    """Milestones outside the run length mean the learning rate never decays."""
    from configs.config import Config

    cfg = Config()
    assert any(m < cfg.epochs for m in cfg.lr_milestones), \
        f"lr_milestones {cfg.lr_milestones} never fire within epochs={cfg.epochs}"
    assert cfg.warmup_epochs < min(cfg.lr_milestones), \
        "warmup should finish before the first decay"


def test_warmup_scheduler():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = WarmupMultiStepLR(optimizer, milestones=(4, 7), gamma=0.1,
                                  warmup_factor=0.1, warmup_iters=3)

    seen = []
    for _ in range(9):
        seen.append(round(optimizer.param_groups[0]["lr"], 5))
        optimizer.step()
        scheduler.step()

    assert seen[0] < seen[1] < seen[2], f"warmup should ramp: {seen[:3]}"
    assert abs(seen[3] - 1.0) < 1e-6, f"should reach base lr after warmup: {seen[3]}"
    assert seen[4] < seen[3] and seen[7] < seen[4], f"step decay missing: {seen}"


def test_trainer_integration():
    """Trainer must unpack (embeddings, logits), apply the identity loss and
    step the scheduler once per epoch."""
    from engine.trainer import Trainer

    model, cfg = make_model("bot", full_ft=True)
    apply_freezing(model, cfg)

    cfg.device = DEVICE
    cfg.epochs = 4
    cfg.val_split = 0.5      # skips the end-of-training checkpoint save
    cfg.world = "open"       # skips the validation pass
    cfg.eval_period = 1000
    cfg.accum_steps = 1

    batches = [
        (torch.zeros(B, T, 3, H, W), mock_labels().cpu(), ["a"] * B, ["v"] * B)
        for _ in range(2)
    ]

    id_loss_calls = []
    seen_lrs = []

    def loss_fn(embeddings, labels, hard_pairs):
        seen_lrs.append(optimizer.param_groups[0]["lr"])
        return embeddings.pow(2).mean()

    def miner(embeddings, labels):
        return None

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1.0
    )
    scheduler = WarmupMultiStepLR(optimizer, milestones=(2,), gamma=0.1,
                                  warmup_factor=0.1, warmup_iters=2)

    trainer = Trainer(model, batches, None, None, optimizer, cfg, loss_fn, miner,
                      scheduler=scheduler)

    original_id_loss = trainer.id_loss_fn
    def counting_id_loss(logits, labels):
        id_loss_calls.append(logits.shape)
        return original_id_loss(logits, labels)
    trainer.id_loss_fn = counting_id_loss

    trainer.train()

    assert len(id_loss_calls) == cfg.epochs * len(batches), \
        f"identity loss applied {len(id_loss_calls)} times, expected {cfg.epochs * len(batches)}"
    assert all(shape == (B, NUM_CLASSES) for shape in id_loss_calls), id_loss_calls

    # One learning rate per epoch; warmup then decay must produce distinct values
    per_epoch = [seen_lrs[i * len(batches)] for i in range(cfg.epochs)]
    assert len(set(per_epoch)) > 1, f"scheduler never changed the learning rate: {per_epoch}"
    assert per_epoch[1] > per_epoch[0], f"warmup did not ramp: {per_epoch}"


TESTS = [
    ("patch shuffle is a permutation", test_shuffle_is_permutation),
    ("bnneck ordering", test_bnneck_ordering),
    ("temporal pooling modes", test_temporal_pool_modes),
    ("bot forward and backward", test_bot_forward_backward),
    ("transreid forward and backward", test_transreid_forward_backward),
    ("transreid requires token backbone", test_transreid_requires_token_backbone),
    ("full fine-tune unfreezes everything", test_full_finetune_unfreezes_everything),
    ("partial fine-tune keeps heads trainable", test_partial_finetune_keeps_heads_trainable),
    ("unfreeze_blocks count", test_unfreeze_blocks_count),
    ("frozen prefix: no_grad split", test_frozen_prefix_saves_memory_without_changing_output),
    ("parameter name prefixes", test_parameter_name_prefixes),
    ("osnet backbone (torchreid)", test_osnet_backbone),
    ("megadescriptor backbones (with/without bot)", test_megadescriptor_backbones),
    ("miewid efficientnet freezing", test_miewid_efficientnet_freezing),
    ("checkpoint classifier keys", test_checkpoint_classifier_keys),
    ("unknown backbone layout raises", test_unknown_backbone_layout_raises),
    ("dinov2 adapter token path", test_dinov2_adapter_token_path),
    ("transforms and clip consistency", test_transforms),
    ("run name round trip", test_run_name_round_trip),
    ("schedule fits epochs", test_schedule_fits_epochs),
    ("warmup scheduler", test_warmup_scheduler),
    ("trainer integration", test_trainer_integration),
]


def main():
    torch.manual_seed(0)
    print(f"device: {DEVICE} | torch {torch.__version__}\n")

    results = []
    for name, fn in TESTS:
        try:
            fn()
            results.append((name, "pass", ""))
        except Exception as exc:  # noqa: BLE001
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))

    width = max(len(name) for name, _, _ in results)
    for name, status, message in results:
        print(f"  [{status:4}] {name:<{width}}  {message}")

    passed = sum(1 for _, status, _ in results if status == "pass")
    print(f"\n{passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
