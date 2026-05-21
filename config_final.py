import os
import warnings
from pathlib import Path

# Suppress annoying scikit-learn warnings when no positive samples are present
warnings.filterwarnings("ignore", message="No positive class found in y_true")

# ── Environment Detection ─────────────────────────────────────────────────────
IS_COLAB = "google.colab" in str(get_ipython()) if "get_ipython" in globals() else False

# ── Global Settings ───────────────────────────────────────────────────────────
SEED = 42

# ── UNet Config ───────────────────────────────────────────────────────────────
UNET_IMAGE_SIZE = 256
UNET_BATCH_SIZE = 16
UNET_EPOCHS = 20
UNET_LR = 2e-4
UNET_VAL_RATIO = 0.15
UNET_P_SYNTH = 0.4
UNET_P_CUTPASTE = 0.3
UNET_P_MULTI_PASTE = 0.3
UNET_USE_SHAPE_BANK = True  # ShapeTextureBank (procedural shapes) vs legacy AnomalyCropBank
UNET_ENCODER = "resnet34"
UNET_USE_ATTENTION_GATES = True
UNET_USE_TTA = True
UNET_MULTIVIEW_AGGREGATE = True
UNET_POSE_JSON = Path(__file__).resolve().parent / "pose_assignments.json"

# ── PatchCore Config ──────────────────────────────────────────────────────────
PC_BACKBONE = "vit_large_patch14_reg4_dinov2.lvd142m"
PC_IMAGE_SIZE = 518
PC_PATCH_SIZE = 14
PC_GRID_SIZE = PC_IMAGE_SIZE // PC_PATCH_SIZE
PC_LAYERS = [7, 12, 16, 20]
PC_PCA_DIM = 512
PC_BATCH_SIZE = 16
PC_CORESET_SIZE = 100_000
PC_KNN_K = 9
PC_SMOOTH_SIGMA = 1.0
PC_CALIB_HIGH = 99
PC_CALIB_N = 256
PC_SCORE_CHUNK = 2048

# ── Ensemble Config ───────────────────────────────────────────────────────────
ENSEMBLE_ALPHAS = [
    round(x, 2) for x in [i / 20 for i in range(21)]
]  # 0.00, 0.05, ..., 1.00
# Strategy for combining UNet + PatchCore. Options:
#   'weighted_sum'    — alpha*unet + (1-alpha)*pc
#   'unet_first_max'  — max(unet, beta*pc)
#   'two_tier'        — rank-tier: any unet>tau outranks every unet<=tau;
#                       PC only orders the lower tier (see fuse_maps).
#   'max', 'residual'
ENSEMBLE_STRATEGY = "weighted_sum"
# Scale grid for PC in 'unet_first_max'. beta=0 → pure UNet, beta=1 → plain max.
ENSEMBLE_BETAS = [round(x, 2) for x in [i / 20 for i in range(21)]]
# Threshold grid for 'two_tier'. tau=0 → almost all FG pixels gated (≈ pure UNet);
# higher tau → more pixels demoted to the PC tier.
ENSEMBLE_TAUS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# PC post-processing (removes contour-ring + fragmented artifacts before fusion)
PC_FG_ERODE_PX = 0  # erode FG mask N pixels inward (kills contour band)
PC_OPEN_RADIUS = 3  # morphological opening kernel half-size (removes thin specks)
PC_MIN_AREA = (
    50  # drop excess-above-p99 connected components smaller than this; 0 disables
)
ENSEMBLE_SMOOTH_SIGMA = 0.9  # Post-fusion smoothing

# ── Paths ─────────────────────────────────────────────────────────────────────
if IS_COLAB:
    ROOT_DIR = Path("/content/drive/MyDrive/ADL/challenge")
    DATASET_DIR = ROOT_DIR / "data" / "adl-2025-2026-anomaly-detection"
    OUTPUT_DIR = ROOT_DIR / "outputs_final"
else:
    # Local paths - assuming we are in the repo root or 'final' subfolder
    CWD = Path.cwd()
    if CWD.name == "final":
        REPO_ROOT = CWD.parent
    else:
        REPO_ROOT = CWD

    ROOT_DIR = REPO_ROOT / "final"
    DATASET_DIR = REPO_ROOT / "dataset" / "adl-2025-2026-anomaly-detection_birefnet"
    OUTPUT_DIR = ROOT_DIR / "outputs"

# UNet specific paths
UNET_MODEL_DIR = OUTPUT_DIR / "unet_models"
UNET_PRED_DIR = OUTPUT_DIR / "unet_preds"
UNET_DESC_CSV = DATASET_DIR / "anomaly_descriptions.csv"

# PatchCore specific paths
PC_BANKS_DIR = OUTPUT_DIR / "patchcore_banks"
PC_SUB_DIR = OUTPUT_DIR / "patchcore_submissions"
PC_HM_DIR = OUTPUT_DIR / "patchcore_heatmaps"

# Ensemble specific paths
ENSEMBLE_DIR = OUTPUT_DIR / "ensemble"


def ensure_dirs():
    for d in [
        UNET_MODEL_DIR,
        UNET_PRED_DIR,
        PC_BANKS_DIR,
        PC_SUB_DIR,
        PC_HM_DIR,
        ENSEMBLE_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
