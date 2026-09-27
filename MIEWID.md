# MiewID Baseline

## What it is
MiewID is a multi-species animal re-identification model released by Wild Me /
Conservation X Labs, used in production for wildlife identification through
the Wildbook platform. It is an EfficientNetV2 backbone trained with an
ArcFace head across dozens of wildlife species (cetaceans, sharks, primates,
big cats, and others), operating on full-body crops rather than faces.

- Model card: https://huggingface.co/conservationxlabs/miewid-msv3
- Project: https://github.com/WildMeOrg/wbia-plugin-miew-id

## Why we add it
- **Full-body input.** Trained on animal body crops, matching our dog
  bounding-box pipeline (unlike face-only baselines).
- **Purpose-built for re-ID.** ArcFace metric-learning objective for
  individual identification, not a repurposed classifier or generic
  self-supervised embedder.
- **Real-world deployment.** Actively used for wildlife ID in the field, so
  it is a credible "off-the-shelf" point of comparison rather than a paper
  artifact.
- **Story-consistent.** MiewID is trained on wildlife, not dogs. Its
  performance on our splits directly supports the thesis that no existing
  animal re-ID foundation model transfers cleanly to the dog domain, which
  motivates the work.

## How it plugs into our pipeline
The builder follows the same contract as the existing backbones
(`dinov2_builder.py`, `swin_builder.py`, `vit_builder.py`):

- Backbone loaded via `transformers.AutoModel.from_pretrained(
  "conservationxlabs/miewid-msv3", trust_remote_code=True)`; the ArcFace
  classification head is not used at inference.
- The embedding dimension is inferred from a dummy forward on init so the
  builder stays robust to checkpoint revisions.
- Temporal attention pool over frames (same module used by every other
  backbone), so video and image inputs are handled uniformly.
- BN-neck + L2 normalization, matching the rest of the pipeline so the
  distance matrices are directly comparable in `evaluation/`.

The HuggingFace checkpoint is cached on first use; no manual download or
weights path is required. `trust_remote_code=True` is needed because the
model card ships a custom forward.

## Reporting
Add MiewID as an additional row in the closed- and open-world tables next to
the frozen-backbone baselines. Because it is pretrained-only (no fine-tuning
on our splits), it belongs in the same section as the other off-the-shelf
comparisons rather than the fine-tuned results.
