import os
import re
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from collections import defaultdict
from PIL import Image

# ── Inference Helpers ─────────────────────────────────────────────────────────

_VIEW_RE = re.compile(r'^(?P<stem>.+)_view(?P<view>[1-5])\.png$')

def parse_view_id(path):
    """Return 0-indexed view from filenames like img_XXXX_viewN.png; 0 if not parseable."""
    m = _VIEW_RE.match(Path(path).name)
    return int(m.group('view')) - 1 if m else 0

# ── PC post-processing ────────────────────────────────────────────────────────

def clean_pc_map(pc_map, fg_mask=None, erode_px=3, open_radius=1):
    """Remove contour-ring + thin-line artifacts from a PatchCore anomaly map.

    1. Eroding the FG mask inward by `erode_px` and multiplying kills the
       ring of contour pixels where PC scores spike due to half-object /
       half-background patches.
    2. Grey-opening with a small kernel removes structures thinner than
       `2*open_radius+1` pixels (specks, hairline contour leftovers).
    """
    from scipy.ndimage import binary_erosion, grey_opening
    out = pc_map.astype(np.float32, copy=True)
    if fg_mask is not None and erode_px > 0:
        eroded = binary_erosion(fg_mask > 0.5, iterations=int(erode_px)).astype(np.float32)
        out = out * eroded
    if open_radius > 0:
        k = 2 * int(open_radius) + 1
        out = grey_opening(out, size=k)
    return out

def remove_small_blobs(score_map, min_area, threshold=0.0):
    """Zero out connected components (pixels above `threshold`) smaller than
    `min_area`. Useful for killing fragmented detections after thresholding."""
    if min_area <= 0:
        return score_map
    from scipy.ndimage import label
    binary = score_map > threshold
    labels, n = label(binary)
    if n == 0:
        return score_map
    sizes = np.bincount(labels.ravel())
    too_small = np.where(sizes < min_area)[0]
    too_small = too_small[too_small != 0]  # keep background label 0 logic intact
    out = score_map.copy()
    out[np.isin(labels, too_small)] = 0.0
    return out

def foreground_mask(unet_mod, img_pil, size):
    """Foreground mask (size, size) from a BiRefNet image's grayscale (>thr=15)."""
    img_rs = img_pil.resize((size, size))
    return unet_mod.foreground_from_image(img_rs)[0].numpy()

def infer_unet_map(unet_mod, model, img_pil, view_id, device, image_size,
                   use_tta=True, fg_mask=None):
    """UNet probability map at (image_size, image_size). If fg_mask is given,
    the background (mask==0) is forced to 0 — see notes on background drift
    caused by loss-masking during training."""
    img_rs = img_pil.resize((image_size, image_size))
    v_t = torch.tensor([view_id], device=device, dtype=torch.long)
    prob = unet_mod._infer_single_tta(model, img_rs, v_t, device, use_tta=use_tta)
    if fg_mask is not None:
        prob = prob * fg_mask
    return prob

def infer_pc_map(pc_mod, extractor, pca, bank, img_path, device,
                 layers, grid_size, out_size, feat_dim=768, fg_mask=None):
    """PatchCore anomaly map upsampled to (out_size, out_size). Optional FG mask."""
    p_in = pc_mod.load_img(img_path).unsqueeze(0).to(device)
    feats = extractor(p_in).cpu().numpy().reshape(-1, len(layers) * feat_dim)
    fp = pca.transform(feats)
    scores = pc_mod.score_patches(fp, bank, device=device)
    hm = pc_mod.make_heatmaps(
        scores.reshape(1, grid_size ** 2), grid_size,
        device=device, out_size=out_size)[0]
    if fg_mask is not None:
        hm = hm * fg_mask
    return hm

# ── Normalization ─────────────────────────────────────────────────────────────

def robust_percentile_norm(m, p50, p99):
    """Contrast-stretch [p50, p99] → [0, 1]. (Legacy; stretches noise floor.)"""
    return np.clip((m - p50) / (p99 - p50 + 1e-8), 0.0, 1.0)

def excess_above_p99(m, p99, clip_to_unit=True):
    """PC-style normalization: signal is only what exceeds the good-image p99.

    Good-pixel scores (typically below p99) map to ~0, strong anomalies (well
    above p99) map toward 1. This keeps noise from being amplified into the
    fusion the way contrast-stretching does.
    """
    out = np.maximum(0.0, m - p99) / (p99 + 1e-8)
    return np.clip(out, 0.0, 1.0) if clip_to_unit else out

def compute_calibration_stats(raw_maps_list):
    """Return (p50, p99) over pooled pixels from the list of good-image maps."""
    all_values = np.concatenate([m.ravel() for m in raw_maps_list])
    p50 = np.percentile(all_values, 50)
    p99 = np.percentile(all_values, 99)
    return p50, p99

# ── Fusion Strategies ─────────────────────────────────────────────────────────

def fuse_maps(unet_map, pc_map, alpha=0.5, strategy='weighted_sum'):
    if strategy == 'weighted_sum':
        return alpha * unet_map + (1.0 - alpha) * pc_map
    elif strategy == 'max':
        return np.maximum(unet_map, pc_map)
    elif strategy == 'unet_first_max':
        # Trust UNet; PatchCore can only fill in where UNet is silent.
        # alpha here is a scale (beta) on the PC map: pc only "wins" a pixel
        # if beta*pc > unet. beta=0 → pure UNet, beta=1 → plain pixel-wise max.
        return np.maximum(unet_map, alpha * pc_map)
    elif strategy == 'two_tier':
        # Rank-tier fusion: every pixel with unet > tau ranks above every
        # pixel with unet <= tau. PC only orders the lower tier.
        # alpha here is tau (UNet gating threshold).
        tau = alpha
        gate = unet_map > tau
        u_tier = np.where(gate, 0.5 + 0.5 * unet_map, 0.0)
        p_tier = 0.5 * pc_map
        return np.maximum(u_tier, p_tier)
    elif strategy == 'residual':
        # Trust PatchCore, use UNet as residual boost
        return np.clip(pc_map + alpha * np.maximum(0, unet_map - 0.2), 0, 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ── Optimization ──────────────────────────────────────────────────────────────

def optimize_weights_per_class(class_name, unet_val_maps, pc_val_maps, gt_masks,
                                alphas, strategy='weighted_sum', verbose=False):
    """Grid search for best alpha on labeled anomalies to maximize *pooled* pixel AP.

    Pools predictions and ground truth across all val images of the class, so
    larger masks contribute proportionally and the AP estimate is less noisy
    than the per-image mean.
    """
    # Resize masks once to match prediction shape
    resized_gt = []
    for fused_ref, gt in zip(unet_val_maps, gt_masks):
        if gt.sum() == 0:
            continue
        if gt.shape != fused_ref.shape:
            gt = np.array(Image.fromarray(gt).resize(
                (fused_ref.shape[1], fused_ref.shape[0]), Image.NEAREST))
        resized_gt.append(gt)

    if not resized_gt:
        return 0.5, 0.0

    gt_flat = np.concatenate([g.ravel().astype(np.uint8) for g in resized_gt])

    best_ap, best_alpha = -1.0, 0.5
    curve = []
    for alpha in alphas:
        fused = [fuse_maps(u, p, alpha, strategy=strategy)
                 for u, p in zip(unet_val_maps, pc_val_maps)]
        pred_flat = np.concatenate([f.ravel() for f, g in zip(fused, gt_masks) if g.sum() > 0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No positive class found in y_true")
            ap = average_precision_score(gt_flat, pred_flat)
        curve.append((float(alpha), float(ap)))
        if ap > best_ap:
            best_ap, best_alpha = ap, alpha

    if verbose:
        print(f"  [{class_name}] alpha vs pooled-pixel-AP:")
        for a, ap in curve:
            marker = ' <-- best' if a == best_alpha else ''
            print(f"    alpha={a:.2f}  AP={ap:.4f}{marker}")

    return best_alpha, best_ap

# ── Dataset Loading ───────────────────────────────────────────────────────────

def get_class_samples(dataset_dir, class_name, split='test'):
    """Helper to get all image paths for a class/split."""
    d = Path(dataset_dir) / class_name / split
    if split == 'train':
        # For calibration we want train/good
        d = d / 'good'
    return sorted(list(d.glob('*.png')))

def load_mask_for_image(img_path, dataset_dir, size=(224, 224)):
    """Load corresponding ground truth mask if it exists."""
    p = Path(img_path)
    # Expected: .../class_XX/train/anomaly_Y/img.png -> .../class_XX/ground_truth_train/anomaly_Y/img.png
    # Or for labeled val pool if we have it
    mask_path = p.parents[2] / 'ground_truth_train' / p.parent.name / p.name
    if mask_path.exists():
        m = np.array(Image.open(mask_path).convert('L').resize(size, Image.NEAREST))
        return (m > 0).astype(np.float32)
    return None

# ── Main Pipeline Logic ───────────────────────────────────────────────────────

def run_ensemble_pipeline(cfg, unet_lib, pc_lib, alphas=None):
    if alphas is None:
        alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = sorted([d.name for d in cfg.DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith('class_')])
    
    best_weights = {}
    submission_results = []
    
    for cls in classes:
        print(f"\n=== Ensembling {cls} ===")
        
        # 1. Load calibration stats for normalization
        # We need raw maps for 'good' validation images from both models
        # For simplicity in this script, we'll assume we can run inference on a few 'good' images
        good_paths = get_class_samples(cfg.DATASET_DIR, cls, 'train')[:32] # Sample 32 for speed
        
        # UNet raw good maps
        # (Assuming unet_lib has helper functions to get raw maps)
        # For now, we use existing functions from the libs
        
        # 2. Optimization on labeled anomalies (if any)
        # Scan train/anomaly folders
        anomaly_root = cfg.DATASET_DIR / cls / 'train'
        val_anomaly_paths = []
        for d in anomaly_root.iterdir():
            if d.is_dir() and d.name.startswith('anomaly_'):
                val_anomaly_paths.extend(list(d.glob('*.png'))[:5]) # Sample for optimization speed
        
        if not val_anomaly_paths:
            print(f"  No labeled anomalies found for {cls}, using alpha=0.5")
            best_alpha = 0.5
        else:
            # Here we would run inference on these val anomalies and optimize
            # To keep this script clean and fast, we'll implement the loop but 
            # in the notebook we'll provide the actual implementation details
            # since full inference takes time.
            best_alpha = 0.5 # Placeholder for logic implemented in notebook
        
        best_weights[cls] = best_alpha
        
        # 3. Final Test Inference & Fusion
        test_paths = get_class_samples(cfg.DATASET_DIR, cls, 'test')
        # (Logic to generate fused maps for test and encode them)
        
    return best_weights

def rle_encode_q8(arr):
    # Reuse PatchCore's efficient RLE
    q      = np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)
    h, w   = q.shape
    flat   = q.T.reshape(-1)
    cuts   = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, flat.size]
    parts  = ['q8rle', str(h), str(w)]
    for val, cnt in zip(flat[starts], ends - starts):
        parts += [str(int(val)), str(int(cnt))]
    return ' '.join(parts)
