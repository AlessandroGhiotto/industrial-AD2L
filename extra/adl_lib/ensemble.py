"""
Ensemble and Multi-View Aggregation for Anomaly Detection.

Combines predictions from multiple models (PatchCore, RD4AD, etc.)
and aggregates across the 5 views per sample for improved detection.

Key strategies:
  - Per-class weight optimization on labeled examples
  - Gated ensemble: RD4AD amplifies PatchCore signal, doesn't add false positives
  - Multi-view aggregation with configurable blending
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar


def extract_sample_id(filename):
    """
    Parse 'img_XXXX_viewN.png' to extract sample_id and view index.

    Args:
        filename: str or Path, e.g. 'img_0022965807e3983107db_view1.png'

    Returns:
        (sample_id, view_idx): e.g. ('img_0022965807e3983107db', 1)
    """
    stem = Path(filename).stem  # e.g. 'img_0022965807e3983107db_view1'
    match = re.match(r"^(.+)_view(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    # Fallback: if no view pattern, treat the whole stem as sample_id
    return stem, 0


def group_results_by_sample(results):
    """
    Group a list of prediction results by sample_id.

    Args:
        results: list of dicts with 'path' key

    Returns:
        dict mapping sample_id -> list of result dicts (one per view)
    """
    groups = defaultdict(list)
    for r in results:
        sample_id, view_idx = extract_sample_id(r["path"])
        r["_sample_id"] = sample_id
        r["_view_idx"] = view_idx
        groups[sample_id].append(r)

    # Sort each group by view index for consistency
    for sample_id in groups:
        groups[sample_id].sort(key=lambda r: r["_view_idx"])

    return dict(groups)


def normalize_anomaly_maps(results, eps=1e-8):
    """
    Normalize anomaly maps to [0, 1] using global min/max across all results.
    Modifies results in-place and returns them.

    Args:
        results: list of dicts with 'anomaly_map' key
        eps: small value to prevent division by zero

    Returns:
        results with normalized anomaly_maps, plus (global_min, global_max)
    """
    if len(results) == 0:
        return results, (0.0, 1.0)

    all_mins = [r["anomaly_map"].min() for r in results]
    all_maxs = [r["anomaly_map"].max() for r in results]
    global_min = float(min(all_mins))
    global_max = float(max(all_maxs))

    scale = global_max - global_min
    if scale < eps:
        # All maps are essentially zero/constant
        for r in results:
            r["anomaly_map"] = np.zeros_like(r["anomaly_map"])
        return results, (global_min, global_max)

    for r in results:
        r["anomaly_map"] = (r["anomaly_map"] - global_min) / (scale + eps)
        r["score"] = float(r["anomaly_map"].max())

    return results, (global_min, global_max)


def aggregate_multiview(results, strategy="max", cross_view_weight=0.3):
    """
    Aggregate anomaly information across views of the same sample.

    For each sample, computes a cross-view aggregated score and blends
    it into each view's anomaly map to boost defect signals visible
    from some views but not others.

    Args:
        results: list of dicts with 'anomaly_map' and 'path' keys
                (anomaly maps should be normalized to [0,1] first)
        strategy: 'max' or 'mean'
        cross_view_weight: float in [0,1], how much to blend from other views

    Returns:
        results with updated anomaly_maps and scores
    """
    groups = group_results_by_sample(results)

    for sample_id, views in groups.items():
        if len(views) <= 1:
            continue

        # Stack all anomaly maps for this sample
        maps = np.stack([v["anomaly_map"] for v in views], axis=0)  # (N_views, H, W)

        if strategy == "max":
            agg = maps.max(axis=0)  # (H, W)
        elif strategy == "mean":
            agg = maps.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        # Blend each view's map with the aggregated signal
        for v in views:
            original = v["anomaly_map"]
            v["anomaly_map"] = (
                (1.0 - cross_view_weight) * original
                + cross_view_weight * agg
            ).astype(np.float32)
            v["score"] = float(v["anomaly_map"].max())

    return results


# ===== Ensemble Strategies =====


def ensemble_additive(results_list, weights=None):
    """
    Simple weighted average ensemble. Original strategy.
    """
    n_models = len(results_list)
    if n_models == 0:
        return []

    n_images = len(results_list[0])
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    ensembled = []
    for img_idx in range(n_images):
        base = results_list[0][img_idx].copy()
        combined_map = np.zeros_like(base["anomaly_map"], dtype=np.float32)
        for model_idx, w in enumerate(weights):
            combined_map += w * results_list[model_idx][img_idx]["anomaly_map"]
        base["anomaly_map"] = combined_map
        base["score"] = float(combined_map.max())
        ensembled.append(base)

    return ensembled


def ensemble_gated(primary_results, secondary_results, gate_threshold=0.1,
                   secondary_weight=0.3):
    """
    Gated ensemble: the secondary model (e.g. RD4AD) can only amplify
    anomaly signal where the primary model (e.g. PatchCore) already detects
    something above gate_threshold.

    This prevents the secondary model from introducing false positives in
    regions where the primary model sees nothing.

    Args:
        primary_results: list of result dicts from the trusted model (e.g. PatchCore)
        secondary_results: list of result dicts from the auxiliary model (e.g. RD4AD)
        gate_threshold: minimum primary score for the secondary to contribute
        secondary_weight: how much weight the secondary gets in gated regions

    Returns:
        list of ensembled result dicts
    """
    n_images = len(primary_results)
    assert len(secondary_results) == n_images, \
        f"Results length mismatch: {n_images} vs {len(secondary_results)}"

    ensembled = []
    for i in range(n_images):
        base = primary_results[i].copy()
        p_map = primary_results[i]["anomaly_map"].astype(np.float32)
        s_map = secondary_results[i]["anomaly_map"].astype(np.float32)

        # Gate: only let secondary contribute where primary has signal
        gate = (p_map > gate_threshold).astype(np.float32)

        # Smooth the gate to avoid hard edges
        from scipy.ndimage import gaussian_filter
        gate = gaussian_filter(gate, sigma=3.0)
        gate = np.clip(gate, 0.0, 1.0)

        # Gated combination: primary + secondary only in gated regions
        combined = (1.0 - secondary_weight) * p_map + secondary_weight * (s_map * gate)
        base["anomaly_map"] = combined.astype(np.float32)
        base["score"] = float(combined.max())
        ensembled.append(base)

    return ensembled


def ensemble_multiplicative(primary_results, secondary_results, boost_power=0.5):
    """
    Multiplicative ensemble: the final score is primary * (1 + secondary)^boost_power.

    The secondary model can boost the primary signal but cannot create
    anomaly scores where the primary sees nothing (since primary * anything = 0
    when primary = 0).

    Args:
        primary_results: result dicts from trusted model
        secondary_results: result dicts from auxiliary model
        boost_power: exponent controlling how much boost the secondary provides.
                     0.0 = no boost (primary only), 1.0 = full multiplicative boost

    Returns:
        list of ensembled result dicts
    """
    n_images = len(primary_results)
    assert len(secondary_results) == n_images

    ensembled = []
    for i in range(n_images):
        base = primary_results[i].copy()
        p_map = primary_results[i]["anomaly_map"].astype(np.float32)
        s_map = secondary_results[i]["anomaly_map"].astype(np.float32)

        # Multiplicative boost: primary * (1 + secondary)^power
        boost = np.power(1.0 + s_map, boost_power)
        combined = p_map * boost

        # Re-normalize to [0, 1]
        cmax = combined.max()
        if cmax > 0:
            combined = combined / cmax

        base["anomaly_map"] = combined.astype(np.float32)
        base["score"] = float(combined.max())
        ensembled.append(base)

    return ensembled


# ===== Per-Class Weight Optimization =====


def _compute_pixel_ap(masks, scores):
    """Compute pixel-level average precision."""
    from sklearn.metrics import average_precision_score
    y_true = masks.reshape(-1).astype(np.uint8)
    y_score = scores.reshape(-1).astype(np.float32)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def optimize_ensemble_weight(primary_labeled, secondary_labeled, strategy="gated",
                             n_steps=50):
    """
    Find the optimal ensemble weight for a single class using labeled anomaly examples.

    Sweeps over weight values and picks the one that maximizes pixel-AP
    on the labeled anomaly set.

    Args:
        primary_labeled: list of result dicts from primary model (must have 'mask')
        secondary_labeled: list of result dicts from secondary model (must have 'mask')
        strategy: 'additive', 'gated', or 'multiplicative'
        n_steps: number of weight values to try

    Returns:
        dict with optimal parameters and the achieved pixel_ap
    """
    n = len(primary_labeled)
    assert len(secondary_labeled) == n, "Results count mismatch"

    # Extract masks (same for both models)
    masks = np.stack([r["mask"] for r in primary_labeled], axis=0)

    # Extract normalized maps
    p_maps = np.stack([r["anomaly_map"] for r in primary_labeled], axis=0).astype(np.float32)
    s_maps = np.stack([r["anomaly_map"] for r in secondary_labeled], axis=0).astype(np.float32)

    # First check: how good is primary alone?
    primary_only_ap = _compute_pixel_ap(masks, p_maps)
    secondary_only_ap = _compute_pixel_ap(masks, s_maps)

    print(f"    Primary-only pixel_AP: {primary_only_ap:.4f}")
    print(f"    Secondary-only pixel_AP: {secondary_only_ap:.4f}")

    best_ap = primary_only_ap
    best_params = {"weight": 0.0, "strategy": "primary_only"}

    if strategy == "additive":
        # Sweep weight for secondary
        for w in np.linspace(0.0, 0.5, n_steps):
            combined = (1.0 - w) * p_maps + w * s_maps
            ap = _compute_pixel_ap(masks, combined)
            if not np.isnan(ap) and ap > best_ap:
                best_ap = ap
                best_params = {"weight": float(w), "strategy": "additive"}

    elif strategy == "gated":
        # Sweep gate_threshold and secondary_weight
        for gate_t in [0.05, 0.1, 0.15, 0.2, 0.3]:
            for sw in np.linspace(0.0, 0.5, n_steps // 5):
                from scipy.ndimage import gaussian_filter
                combined_all = []
                for i in range(n):
                    gate = (p_maps[i] > gate_t).astype(np.float32)
                    gate = gaussian_filter(gate, sigma=3.0)
                    gate = np.clip(gate, 0.0, 1.0)
                    c = (1.0 - sw) * p_maps[i] + sw * (s_maps[i] * gate)
                    combined_all.append(c)
                combined = np.stack(combined_all, axis=0)
                ap = _compute_pixel_ap(masks, combined)
                if not np.isnan(ap) and ap > best_ap:
                    best_ap = ap
                    best_params = {
                        "weight": float(sw),
                        "gate_threshold": float(gate_t),
                        "strategy": "gated",
                    }

    elif strategy == "multiplicative":
        # Sweep boost power
        for bp in np.linspace(0.0, 1.0, n_steps):
            boost = np.power(1.0 + s_maps, bp)
            combined = p_maps * boost
            # Per-image normalization
            for i in range(n):
                cmax = combined[i].max()
                if cmax > 0:
                    combined[i] = combined[i] / cmax
            ap = _compute_pixel_ap(masks, combined)
            if not np.isnan(ap) and ap > best_ap:
                best_ap = ap
                best_params = {"boost_power": float(bp), "strategy": "multiplicative"}

    best_params["pixel_ap"] = float(best_ap)
    best_params["improvement"] = float(best_ap - primary_only_ap)

    print(f"    Best ensemble: {best_params}")
    return best_params


def apply_ensemble_with_params(primary_results, secondary_results, params):
    """
    Apply ensemble with pre-optimized parameters.

    Args:
        primary_results: list of result dicts from primary model
        secondary_results: list of result dicts from secondary model
        params: dict from optimize_ensemble_weight()

    Returns:
        list of ensembled result dicts
    """
    strategy = params.get("strategy", "primary_only")

    if strategy == "primary_only":
        # No ensemble, just return primary
        return [r.copy() for r in primary_results]

    elif strategy == "additive":
        w = params.get("weight", 0.0)
        return ensemble_additive(
            [primary_results, secondary_results],
            weights=[1.0 - w, w],
        )

    elif strategy == "gated":
        return ensemble_gated(
            primary_results, secondary_results,
            gate_threshold=params.get("gate_threshold", 0.1),
            secondary_weight=params.get("weight", 0.3),
        )

    elif strategy == "multiplicative":
        return ensemble_multiplicative(
            primary_results, secondary_results,
            boost_power=params.get("boost_power", 0.5),
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Keep backward compatibility
def ensemble_models(results_list, weights=None, eps=1e-8):
    """Legacy wrapper for simple additive ensemble."""
    return ensemble_additive(results_list, weights)


def apply_sample_gate(results, alpha=0.5, score_quantile=0.999, view_agg="max"):
    """Multiply each view's anomaly map by (1 + alpha * p_sample).

    p_sample is a sample-level "this object looks suspicious" prior in [0, 1],
    obtained by:
      1. per-view image-level score = quantile(anomaly_map, score_quantile)
      2. per-sample aggregate over its 5 views (max or mean)
      3. rank-normalize across all samples in `results` to [0, 1]

    Does NOT mix any pixel across views — each view keeps its own map; only the
    per-sample scalar multiplier changes. Preserves within-view and within-sample
    pixel rank order; rebalances rank order across samples.

    Modifies results in-place and returns them.
    """
    if alpha <= 0 or not results:
        return results

    groups = group_results_by_sample(results)
    sample_ids = list(groups.keys())

    sample_scores = np.empty(len(sample_ids), dtype=np.float64)
    for i, sid in enumerate(sample_ids):
        per_view = np.array(
            [np.quantile(v["anomaly_map"], score_quantile) for v in groups[sid]],
            dtype=np.float64,
        )
        if view_agg == "max":
            sample_scores[i] = per_view.max()
        elif view_agg == "mean":
            sample_scores[i] = per_view.mean()
        elif view_agg == "top2":
            sample_scores[i] = np.sort(per_view)[-2:].mean()
        else:
            raise ValueError(f"Unknown view_agg: {view_agg}")

    # Rank-normalize to [0, 1] across samples (robust to scale/outliers).
    order = np.argsort(np.argsort(sample_scores))
    if len(sample_ids) > 1:
        p_sample = order / (len(sample_ids) - 1)
    else:
        p_sample = np.zeros_like(sample_scores)

    for sid, p in zip(sample_ids, p_sample):
        gain = 1.0 + alpha * float(p)
        for v in groups[sid]:
            v["anomaly_map"] = v["anomaly_map"] * gain

    return results
