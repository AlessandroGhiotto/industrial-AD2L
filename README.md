# Industrial Anomaly Detection — ADL 2025-2026

Final submission for the ADL 2025-2026 Industrial Visual Anomaly Detection competition.
Given five product classes with five camera views each, the task is to detect and localize
defects at pixel level.

The pipeline combines an unsupervised memory-based detector (PatchCore on DINOv2 features)
with a supervised segmentation model (U-Net trained on synthetic anomalies) and fuses them
at the pixel level.

## Pipeline overview

The five steps below mirror the order in which the notebooks should be run.

### 1. `preprocess_birefnet_dataset.py` — background removal

Industrial images come with cluttered backgrounds (conveyor belts, fixtures, shadows) that
hurt both PatchCore's memory bank and U-Net's segmentation signal. We preprocess the whole
dataset with [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) to mask out the background
and replace it with a constant fill, keeping only the object foreground.

```bash
python preprocess_birefnet_dataset.py \
  --input-root  ./dataset/adl-2025-2026-anomaly-detection \
  --output-root ./dataset/adl-2025-2026-anomaly-detection_birefnet \
  --fill black
```

All downstream notebooks consume the cleaned `_birefnet` dataset.

### 2. `pose_clustering.ipynb` — viewpoint clustering

The filename suffix (`view1`, `view2`, …) does **not** consistently encode the same camera
angle across different physical samples. To recover pose-coherent groups we mean-pool DINOv2
patch tokens into a 768-d global descriptor per image, L2-normalize, and run K-means per
class with a silhouette sweep to pick K. The resulting per-image pose label is written to
`pose_assignments.json` and reused by both PatchCore (pose-coherent memory banks) and U-Net
(pose-aware CutPaste augmentation — see below).

### 3. `PatchCore.ipynb` — PatchCore with DINOv2

Unsupervised, memory-based normality prior:

- **Backbone**: DINOv2 ViT-L/14 (`vit_large_patch14_reg4_dinov2.lvd142m`) at 518×518.
- **Feature aggregation**: tokens from layers 7, 12, 16, 20 are concatenated to capture
  multi-scale semantics.
- **Dim reduction**: GPU-based PCA down to 512 components.
- **Coreset**: greedy farthest-point sub-sampling of the patch bank (≤100k anchors).
- **Scoring**: k-NN (k=9) distance against the coreset, smoothed with a Gaussian (σ=1.0)
  and globally calibrated per class using the 99th percentile.

### 4. `Unet.ipynb` — supervised U-Net on synthetic anomalies

Complementary supervised branch trained purely on **synthetic** defects pasted onto good
images:

- **Architecture**: U-Net with a `resnet34` encoder and **attention gates** on the skip
  connections.
- **Synthetic anomalies**: a mix of procedural ShapeTextureBank shapes, real anomaly crops
  (CutPaste with geometric + photometric jitter), and multi-paste compositions.
- **Pose-aware CutPaste**: anomaly crops are only pasted onto goods that share the same
  pose cluster (from step 2); anomalous-train images are assigned a pose via a 32×32
  grayscale nearest-neighbour lookup against the labeled goods.
- **Inference**: test-time augmentation (TTA) and per-sample multi-view aggregation across
  the 5 views of a physical object.

### 5. `Ensemble.ipynb` — pixel-level fusion

Final fusion of the two complementary signals:

1. **Normalization** — robust per-class percentile scaling so the two maps live on the
   same scale.
2. **Weight search** — per-class α swept on labeled validation anomalies to maximize
   pixel-AP.
3. **Fusion** — pixel-level combination (default: `weighted_sum`; alternatives include
   `unet_first_max` and a `two_tier` rank-based scheme). PatchCore is further cleaned with
   morphological opening / connected-component filtering before being mixed in.

The submission CSV (q8rle-encoded anomaly maps) is written from this notebook.

## Repo layout

```
final/
├── preprocess_birefnet_dataset.py   # step 1
├── pose_clustering.ipynb            # step 2
├── PatchCore.ipynb                  # step 3
├── Unet.ipynb                       # step 4
├── Ensemble.ipynb                   # step 5
├── pose_assignments.json            # output of step 2 (committed for reproducibility)
├── src/                             # python modules imported by the notebooks
│   ├── config_final.py              # single source of truth for paths/hyperparameters
│   ├── birefnet.py                  # BiRefNet wrapper used by step 1
│   ├── patchcore_final.py           # DINOv2 + PCA + coreset + k-NN
│   ├── unet_final.py                # attention U-Net + synthetic anomaly augmentations
│   ├── ensemble_final.py            # normalization, weight search, fusion strategies
│   └── visualize_final.py           # qualitative PDF reports
├── outputs/                         # checkpoints, banks, heatmaps, submissions (gitignored)
└── extra/                           # exploratory work — see below
```

## Running on Colab

Each of the three main notebooks (`PatchCore.ipynb`, `Unet.ipynb`, `Ensemble.ipynb`) starts
with a **Colab Setup** cell that, when run on Colab:

1. clones this repo into `/content/industrial-AD2L`,
2. `cd`s into it,
3. `pip install -q timm gdown`,
4. downloads and unzips the BiRefNet-preprocessed dataset from Google Drive.

Just open the notebook on Colab and run the first cell — everything else falls into place.
Running locally is a no-op for that cell.

## Extra — things we tried shallowly

The `extra/` folder collects alternative models and experiments that we benchmarked but did
not include in the final ensemble. They are kept here as a record of the design space we
explored.

Models under `extra/adl_lib/` (with matching notebooks in `extra/notebooks/` and training
scripts in `extra/scripts/`):

- **DRAEM** — discriminative reconstruction with a synthetic anomaly generator.
- **EfficientAD** — student–teacher PDN + autoencoder.
- **RD4AD** — reverse distillation: frozen teacher, trainable bottleneck + decoder,
  cosine-distance scoring.
- **SimpleNet** — feature-space discriminator trained against Gaussian-noised normal
  features used as fake anomalies.
- **Student–Teacher** — classic feature-distillation baseline (Wide-ResNet-50-2 teacher,
  ResNet-18 student).
- **WinCLIP+** — zero-shot CLIP-based detector with windowed prompts.

Other notebooks under `extra/`:

- `autoenc1-2.ipynb` — autoencoder baselines.
- `ensemble_benchmark.ipynb` — sweeping ensemble strategies across model pairs.
- `tentativo_fusione_2_layer_.ipynb` — two-layer PatchCore fusion attempt.
- `tentativo_texture_free.ipynb` — texture-suppression preprocessing attempt.
- `tentativo_uso_cls_token.ipynb` — using the DINOv2 CLS token instead of patch tokens.

## Repository

Public repo: https://github.com/AlessandroGhiotto/industrial-AD2L
