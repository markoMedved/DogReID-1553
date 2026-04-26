# Models: Architecture & Strategy

## 1. Shared Custom Pipeline Additions
Regardless of the backbone, all three models utilize a unified pipeline designed for memory-efficient video processing and metric learning:
* **Chunked Forward Pass:** Video frames are flattened and processed in chunks to prevent GPU Out-of-Memory (OOM) errors.
* **Temporal Attention Pooling:** A custom MLP replaces naive frame averaging by learning to weigh frames dynamically (e.g., prioritizing clear shots while ignoring blurry or occluded frames).
* **BN-Neck:** A 1D Batch Normalization layer (with a frozen bias) is appended to stabilize the embedding distribution, which is highly beneficial for Triplet Margin Loss.
* **L2 Normalization:** Final embeddings are forced onto a unit hypersphere, making cosine similarity equivalent to a simple dot product for distance calculations.
* **Cross-Task Compatibility (Image-to-Video):** For the specific Image-to-Video Re-ID task, these exact same models (trained strictly on video datasets) are utilized without modification.

## 2. Shared Fine-Tuning Strategy (Layer Management)
To prevent catastrophic forgetting of pre-trained vision features while safely adapting to the Re-ID domain, all models share this baseline training strategy:
* **Global Freeze:** All backbone weights are initially entirely frozen.
* **Differential Learning Rates:** Unfrozen backbone layers are trained at a **10x smaller learning rate** than the custom head to ensure gentle domain adaptation.
* **Surgical Unfreezing:** Only the deepest, high-level semantic layers are unfrozen (specifics detailed per model below).

## 3. Data Augmentation Strategy (Transforms)
To ensure the models generalize well to real-world, uncontrolled environments, an augmentation pipeline is applied at the frame level.

* **Training Transforms (Robustness & Regularization):**
  * **Spatial Variance:** `RandomResizedCrop` and `RandomHorizontalFlip` simulate different camera distances, angles, and subject orientations.
  * **Lighting Variance:** `ColorJitter` (adjusting brightness, contrast, saturation, and hue) ensures the model doesn't overfit to specific lighting conditions or camera sensor quirks.
  * **Occlusion Simulation:** `RandomErasing` randomly drops out patches of the image. This is a critical regularizer for Re-ID, forcing the model to learn global canine features rather than memorizing a single distinct marker that might be occluded in reality.
  * **Standardization:** ImageNet `Normalize` is applied to align the inputs with the pre-trained backbone distributions.
  * **Transforms:**  (`Resize` to 256, `CenterCrop` to 224) 

## 4. Metric Learning Objective (Loss & Mining)
The network does not use standard classification loss (like Cross-Entropy) because the identities in the test set do not exist in the training set. Instead, it relies on distance-based metric learning.

* **Triplet Margin Loss:** The core objective optimizes the relative distance between embeddings. It pulls embeddings of the same dog (Anchor & Positive) closer together while pushing embeddings of different dogs (Anchor & Negative) apart by a minimum defined margin.
* **Hard Pair Mining:** The pipeline utilizes a **Miner** to actively select the hardest positives (the most dissimilar frames of the *same* dog) and the hardest negatives (the most visually similar frames of *different* dogs) within each batch. 

## 5. Shared Training & Optimization Configuration
* **PK Batch Sampling ($P \times K$):** `batch_size = 16`, `k = 4`. Every batch contains exactly 4 distinct dog identities ($P$), with 4 video clips per dog ($K$). This guarantees valid positive and negative pairs for Triplet Loss computation.
* **Video Constraints:** `clip_len = 16`. We sample 16 frames uniformly from each video.
* **Memory Management:** `chunk_size = 16`. Processing video transformers is VRAM-intensive, so frames are pushed through the backbone in controlled chunks.
* **Gradient Accumulation:** `accum_steps = 8`. To achieve a stable optimization step without OOM errors, gradients are accumulated to simulate an effective batch size of 128 (16 * 8) before updating weights.
* **Epochs:** Models are trained for `epochs = 50`, although the loss dropping mostly finished earlier
* **Dataloading:**`num_workers = 12` to prevent data loading bottlenecking.
* **Embedding Dimensions:** Standardized to `embedding_dim = 768` (Note: automatically scales to `1024` specifically for the SwinV2 architecture).
* **Optimizer:** **AdamW** is used to properly decouple weight decay from the gradient updates.
* **Learning Rate Scheduler:** To prevent the randomly initialized custom Temporal Pooling MLP from sending destructive gradients into the newly unfrozen backbone layers, a **Linear Warmup** phase is typically employed for the initial epochs, allowing the custom head to stabilize before standard decay begins.
* **Hyperparameter Tuning Strategy:** The core optimization variables (`lr`, `margin`, `weight_decay`) for each model were determined through systematic hyperparameter tuning using a 20% validation split strictly on the **Closed-Set** split (We use the same parameters for open world). The chosen combinations achieved the **highest average mean Average Precision (mAP) over the final 10 epochs**.

## 6. Evaluation & Inference Protocol (Closed vs. Open World)

### Task Definitions
* **Closed-World (Ranking):** Every dog in the query set is guaranteed to exist in the gallery. The system's objective is pure ranking—sorting the gallery to ensure the correct matches appear at the top.
* **Open-World (Identification + Rejection):** A more realistic scenario where some query dogs *do not* exist in the gallery. The system performs verification thresholding. It must find the closest gallery match and decide, based on a distance threshold, whether to identify the dog (if the distance is below the threshold) or reject it as an "unknown" dog (if the distance is above the threshold).

### Step-by-Step Inference & CSV Construction
1. **Feature Extraction:** The model processes all Query and Gallery dataloaders in evaluation mode, mapping outputs to L2-normalized unit embeddings.
2. **Distance Matrix Computation:** A pairwise L2 distance matrix is calculated (`torch.cdist`) between all query embeddings and all gallery embeddings.
3. **CSV Standardization:** The matrix is exported directly to a comma-separated (`,`) CSV file where rows represent exact Query IDs, columns represent Gallery IDs, and values are the resulting L2 distances. This standardizes the required ordering for external benchmarking. More details at https://project-puppies.com/leaderboard/submit.

### Measuring Uncertainty (Bootstrapping)
For all metrics we estimate uncertainty by bootstrapping identities with 100 resamples. We report
95% confidence intervals for the point-estimates in the tables and point-wise 95% confidence bands for the curves in the figures

### Metrics
* **Closed-World Metrics:**
  * **mAP (mean Average Precision):** Measures the overall robustness of the retrieval (finding *all* correct matches in the gallery).
  * **CMC (Cumulative Match Characteristic):** curve, Rank 1, Rank 5
* **Open-World Metrics:** Evaluated across a sweep of 500 thresholds to generate a smooth curve.
  * **DIR (Detection and Identification Rate):** The percentage of known queries correctly matched to their true identity below a given threshold.
  * **FAR (False Alarm Rate):** The percentage of unknown queries incorrectly assigned a gallery identity (failing to reject). We explicitly report DIR at strict FARs (1%, 5%, 10%).

---

## 7. Model Profiles

### DINOv2
A self-supervised model trained on a very large dataset. 

* **Why DINOv2:** Chosen because its self-supervised training naturally isolates subjects from complex backgrounds, its "registers" effectively absorb visual noise, and it produces the exceptionally dense features necessary for distinguishing fine-grained micro-textures.
* **Specific Unfreezing:** Only the **last two transformer blocks** and the **final layer normalization** are unfrozen to adapt high-level semantic logic to canine features.
* **Tuned Hyperparameters:**
  * **Learning Rate (`lr`):** `5e-05`
  * **Triplet Margin (`margin`):** `0.3`
  * **Weight Decay (`weight_decay`):** `1e-05`

### SwinV2
A hierarchical Vision Transformer that computes self-attention locally using shifted windows.

* **Why SwinV2:** Selected for its hierarchical architecture and shifted window attention, which efficiently compute multi-scale, high-dimensional embeddings that are scale-invariant and capture both fine local details and broad global semantics.
* **Specific Unfreezing:** The **entire final hierarchical stage** and the **final layer normalization** are unfrozen to adapt the highest-level semantic representations to the specific domain.
* **Tuned Hyperparameters:**
  * **Learning Rate (`lr`):** `2e-05`
  * **Triplet Margin (`margin`):** `0.25`
  * **Weight Decay (`weight_decay`):** `5e-05`

### ViT
A standard supervised Vision Transformer architecture acting as a reliable, balanced baseline.

* **Why ViT:** Utilized as a proven, well-understood baseline that balances computational cost with feature granularity while inherently capturing long-range spatial dependencies through standard global self-attention.
* **Specific Unfreezing:** Only the **last two transformer blocks** and the **final layer normalization** are unfrozen to adapt high-level semantic logic to the specific domain.
* **Tuned Hyperparameters:**
  * **Learning Rate (`lr`):** `5e-05`
  * **Triplet Margin (`margin`):** `0.3`
  * **Weight Decay (`weight_decay`):** `1e-05`