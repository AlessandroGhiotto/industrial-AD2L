import random

import numpy as np
import torch

# ===== GENERAL SETTINGS =====
SEED = 7
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0
ENABLE_IMAGE_CACHE = True

PATH = "./dataset/adl-2025-2026-anomaly-detection_birefnet"
CLASS_NAME = "class_01"

# ===== DATA & VALIDATION SETTINGS =====
VAL_GOOD_RATIO = 0.15
BATCH_SIZE = 32

# ===== ANOMALY MAP POSTPROCESSING =====
ANOMALY_MAP_SIGMA = 1.0
ANOMALY_MAP_BACKGROUND_PERCENTILE = 10.0

# ===== PATCHCORE BACKBONE & FEATURES =====
PATCHCORE_BACKBONE_CANDIDATES = ["vit_large_patch14_reg4_dinov2.lvd142m", "wide_resnet50_2"]
PATCHCORE_OUT_INDICES = (17, 23)
PATCHCORE_PATCHSIZE = 3

# ===== PATCHCORE MEMORY BANK & INDEXING =====
PATCHCORE_CORESET_FRACTION = 0.2
PATCHCORE_CANDIDATE_POOL_SIZE = 300_000
PATCHCORE_RANDOM_PROJECTION_DIM = 128
PATCHCORE_BANK_CHUNK_SIZE = 2048

# ===== FAISS INDEX PARAMETERS =====
FAISS_MAX_BANK_SIZE = 60_000
FAISS_NPROBE = 32
FAISS_GPU_SAFE_MAX_M = 32
FAISS_N_LIST_MIN = 128
FAISS_N_LIST_MAX = 512
FAISS_TRAIN_SIZE_FACTOR = 64
APPROX_SEARCH = False

# ===== NORMALIZATION & BACKBONE =====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ===== VISUALIZATION PARAMETERS =====
HEATMAP_OVERLAY_ALPHA = 0.45
SCORE_DISTRIBUTION_BINS = 20
SHOW_DETECTION_N = 5
Q8RLE_SCALE = 255

# ===== PDF EXPORT PARAMETERS =====
EXPORT_PDF_PATH = "./artifacts/patchcore/all_classes_gt_predictions.pdf"
EXPORT_DPI = 140
EXPORT_MAX_IMAGES_PER_CLASS = (
    None  # set an int to limit images per class, or None for all
)
EXPORT_FORCE_REBUILD = (
    False  # True = always fit model; False = try loading saved FAISS index first
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===== ANOMALY MAP CLEANING =====
POSTPROCESS_CLEANING_THRESHOLD = 75.0  # percentile for core binarization
POSTPROCESS_MIN_AREA_RATIO = 0.2      # keep blobs > 20% of largest blob area

# ===== RD4AD SETTINGS =====
RD4AD_BACKBONE = "wide_resnet50_2"
RD4AD_OUT_INDICES = (1, 2, 3)
RD4AD_BOTTLENECK_DIM = 256
RD4AD_LR = 5e-4
RD4AD_WEIGHT_DECAY = 1e-4
RD4AD_EPOCHS = 100
RD4AD_EARLY_STOP_PATIENCE = 15
RD4AD_ANOMALY_MAP_SIGMA = 4.0

# ===== ENSEMBLE SETTINGS =====
ENSEMBLE_PATCHCORE_WEIGHT = 0.4
ENSEMBLE_RD4AD_WEIGHT = 0.6
