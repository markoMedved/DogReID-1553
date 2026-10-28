# ECCV-26-SCL

Code release for cross-species animal re-identification with Semantic Consistency Learning (SCL). The released implementation centers on the `MetaN` training package and combines:

- a TransReID-style ViT backbone (`vit_base_patch16_224_TransReID`);
- SCFG frequency normalization inserted at ViT blocks `[0, 4, 8]`;
- NFC reciprocal-neighbor consistency with an EMA teacher and feature queue;
- cross-species reciprocal-neighbor alignment for animal identity transfer.

This repository is intentionally compact for open-source release. Ablation-only code, visualization scripts, local machine scripts, experiment outputs, checkpoints, and private datasets are not included.

## Repository Layout

```text
config/                         Default configuration definitions
configs/                        Main SCL/MetaN experiment config
datasets/                       WildlifeReID-10K and combined dataset loaders
loss/                           ID, triplet, and NFC losses
model/                          TransReID ViT backbone and SCFG modules
processor/                      Training and validation loops
scripts/training/               Reproducible launch scripts
solver/                         Optimizer and learning-rate scheduler helpers
utils/                          Logging, metrics, IO, and evaluation utilities
train.py                        Training entry point
test.py                         Evaluation entry point
requirements.txt                Python package requirements
```

## Installation

Create an environment and install the Python dependencies:

```bash
conda create -n scl-animal-reid python=3.10 -y
conda activate scl-animal-reid
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your machine if the default wheel selected by `pip` is not appropriate.

## Data Preparation

The default training config expects WildlifeReID-10K metadata at:

```text
data/WildlifeReID-10K/metadata.csv
```

The loader also accepts a root directory that directly contains `metadata.csv`. Image paths are resolved from the metadata table by `datasets/wildlifereid10k.py`.

For local runs, keep data outside the repository or under `data/`; large files are ignored by `.gitignore`.

## Pretrained Weights

The default config expects the ImageNet-pretrained ViT checkpoint at:

```text
pretrained/jx_vit_base_p16_224-80ecf9dd.pth
```

You can override both the dataset root and pretrained checkpoint path from the launch script:

```bash
DATA_ROOT=/path/to/WildlifeReID-10K \
PRETRAIN_PATH=/path/to/jx_vit_base_p16_224-80ecf9dd.pth \
bash scripts/training/train_w10k29_to_full12_scfg.sh 0 outputs/scl_metan
```

## Training

Run the default WildlifeReID-10K training recipe:

```bash
bash scripts/training/train_w10k29_to_full12_scfg.sh
```

The script accepts optional positional arguments:

```bash
bash scripts/training/train_w10k29_to_full12_scfg.sh GPUS OUTPUT_DIR EPOCHS BATCH_SIZE
```

Example multi-GPU launch:

```bash
DATA_ROOT=/data/WildlifeReID-10K \
PRETRAIN_PATH=/models/jx_vit_base_p16_224-80ecf9dd.pth \
bash scripts/training/train_w10k29_to_full12_scfg.sh 0,1 outputs/scl_metan 60 128
```

The script writes the resolved launch options to `OUTPUT_DIR/launch_command.txt` and starts distributed training with `torch.distributed.run`.

## Main Configuration

The main release configuration is:

```text
configs/transreid_nfc_scfg_w10k29_to_full12.yml
```

Important defaults include:

- `MODEL.TRANSFORMER_TYPE: vit_base_patch16_224_TransReID`
- `CHANGE.METHODS.FREQ_NORM_KIND: scfg`
- `CHANGE.METHODS.FREQ_NORM_POSITIONS: [0, 4, 8]`
- `CHANGE.METHODS.SCFG_CONSTRAINED: True`
- `CHANGE.METHODS.NFC_TRAINING: True`
- `CHANGE.METHODS.NFC_QUEUE_SIZE: 4096`
- `CHANGE.METHODS.NFC_EMA_MOMENTUM: 0.993`
- standard NFC neighbors: `K1=8`, `K2=8`
- cross-species NFC neighbors: `K1=4`, `K2=4`, weight `3.0`, margin `1.0`, cosine metric

Configuration values can be overridden from the command line after the config path:

```bash
python train.py \
  --config_file configs/transreid_nfc_scfg_w10k29_to_full12.yml \
  OUTPUT_DIR outputs/debug_run \
  SOLVER.MAX_EPOCHS 5 \
  SOLVER.IMS_PER_BATCH 32
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
python test.py \
  --config_file configs/transreid_nfc_scfg_w10k29_to_full12.yml \
  --model_path outputs/scl_metan/transformer_60.pth
```

You can override test dataset names and roots through command-line config options when using combined evaluation sets:

```bash
python test.py \
  --config_file configs/transreid_nfc_scfg_w10k29_to_full12.yml \
  --model_path outputs/scl_metan/transformer_60.pth \
  DATASETS.TEST_COMBINE_NAMES "['panda','elephant']" \
  DATASETS.TEST_ROOTS "['/path/full12/panda','/path/full12/elephant']"
```

## Notes

- Training artifacts, checkpoints, logs, CSV files, and dataset folders are ignored by default.
- The public package keeps the core SCL/MetaN training path and removes private analysis utilities.
- `test.py` is a lightweight evaluation entry; adapt the config overrides to match your benchmark layout.

## License

This project is released under the MIT license. See `LICENSE` for details.
