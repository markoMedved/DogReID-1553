"""Torchreid model-zoo checkpoints for OSNet, and how to fetch them.

Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html

Which checkpoint to use
-----------------------
Dogs are an unseen domain for every one of these models, so the relevant
criterion is cross-domain generalization, not same-domain accuracy. That rules
out the strongest same-domain checkpoint (osnet_x1_0 on Market1501, 94.2 Rank-1)
precisely because it is specialized to one domain.

The default below is the OSNet-AIN multi-source domain-generalization model:

- AIN is the variant introduced in the TPAMI'21 extension specifically to
  improve generalization to unseen domains, by adding instance normalization.
- It is trained on three source datasets (MSMT17 + DukeMTMC + CUHK03) under the
  authors' multi-source domain-generalization protocol.
- It is the best model on three of the four targets in that protocol.
- The zoo evaluates AIN models with cosine distance, which is what
  evaluation/ uses.

The held-out target in the original protocol (Market1501) does not matter here,
since dogs are unseen by all of these models regardless.

Downloading
-----------
The files are on Google Drive, so fetch them on a login node:

    pip install gdown
    python -m models.osnet_weights --name osnet_ain_ms_d_c

then point cfg.osnet_weights at the resulting file.

Note on augmentation
--------------------
The model zoo trains these with `random_flip` and `color_jitter` and explicitly
avoids random erasing, on the grounds that heavy augmentation can harm
cross-dataset generalization. Our BoT pipeline uses random erasing at p=0.5, so
consider `re_prob = 0.0` when fine-tuning from these weights, and report which
was used.
"""

import argparse
import os

# name -> (google drive file id, osnet variant, description)
WEIGHTS = {
    # Multi-source domain generalization (Zhou et al., TPAMI 2021)
    "osnet_ain_ms_d_c": (
        "1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB", "osnet_ain_x1_0",
        "AIN, trained on MSMT17+Duke+CUHK03, 73.3 (45.8) on held-out Market1501",
    ),
    "osnet_ain_ms_m_c": (
        "1YjJ1ZprCmaKG6MH2P9nScB9FL_Utf9t1", "osnet_ain_x1_0",
        "AIN, trained on MSMT17+Market+CUHK03, 65.6 (47.2) on held-out Duke",
    ),
    "osnet_ibn_ms_d_c": (
        "14sH6yZwuNHPTElVoEZ26zozOOZIej5Mf", "osnet_ibn_x1_0",
        "IBN, trained on MSMT17+Duke+CUHK03, 73.0 (44.9) on held-out Market1501",
    ),
    "osnet_ms_d_c": (
        "1tuYY1vQXReEd8N8_npUkc7npPDDmjNCV", "osnet_x1_0",
        "Base OSNet, trained on MSMT17+Duke+CUHK03, 72.5 (44.2) on held-out Market1501",
    ),
    # Single-source cross-domain, simpler to describe in a paper
    "osnet_ain_msmt17": (
        "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal", "osnet_ain_x1_0",
        "AIN, trained on MSMT17 (combineall), 70.1 (43.3) -> Market, 71.1 (52.7) -> Duke",
    ),
    # Same-domain, included for completeness; specialized, weakest transfer
    "osnet_market1501": (
        "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA", "osnet_x1_0",
        "Base OSNet, trained on Market1501, 94.2 (82.6) same-domain",
    ),
}

DEFAULT = "osnet_ain_ms_d_c"


def download(name: str = DEFAULT, out_dir: str = "pretrained") -> str:
    """Fetch a checkpoint with gdown and return its path."""
    if name not in WEIGHTS:
        raise ValueError(f"Unknown weights {name!r}; available: {sorted(WEIGHTS)}")

    file_id, variant, desc = WEIGHTS[name]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")

    if os.path.exists(path):
        print(f"[osnet_weights] already present: {path}")
        return path

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Downloading requires gdown: pip install gdown") from exc

    print(f"[osnet_weights] {name}: {desc}")
    gdown.download(id=file_id, output=path, quiet=False)
    print(f"[osnet_weights] set cfg.osnet_variant = {variant!r}")
    print(f"[osnet_weights] set cfg.osnet_weights = {path!r}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSNet weights")
    parser.add_argument("--name", default=DEFAULT, choices=sorted(WEIGHTS))
    parser.add_argument("--out_dir", default="pretrained")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    args = parser.parse_args()

    if args.list:
        for key, (_, variant, desc) in WEIGHTS.items():
            marker = " (default)" if key == DEFAULT else ""
            print(f"{key:20} {variant:16} {desc}{marker}")
    else:
        download(args.name, args.out_dir)
