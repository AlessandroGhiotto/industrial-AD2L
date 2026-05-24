"""Shared evaluation helper for the methods comparison table.

The protocol mirrors the one used in `ensemble_final.optimize_weights_per_class`:
all labeled `train/anomaly_*` images of a class are scored, every GT mask is
resized to match the prediction, then predictions and masks are pooled and a
single pixel-AP is computed per class.

Any anomaly detector can be plugged in by supplying a callable
`predict_fn(img_path, img_pil) -> np.ndarray  # shape (H, W)`.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

PredictFn = Callable[[Path, Image.Image], np.ndarray]


def _list_labeled_anomalies(dataset_dir: Path, class_name: str) -> List[Dict]:
    """Return list of dicts {img, mask, anomaly_type} for every labeled anomaly."""
    cls_root = Path(dataset_dir) / class_name
    train_root = cls_root / "train"
    gt_root = cls_root / "ground_truth_train"
    items: List[Dict] = []
    if not train_root.exists():
        return items
    for d in sorted(train_root.iterdir()):
        if not (d.is_dir() and d.name.startswith("anomaly_")):
            continue
        mask_dir = gt_root / d.name
        for img_path in sorted(d.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue
            items.append({
                "img": img_path,
                "mask": mask_path,
                "anomaly_type": d.name,
            })
    return items


def pooled_pixel_ap_for_class(
    predict_fn: PredictFn,
    class_name: str,
    dataset_dir: Path,
    progress: bool = True,
) -> float:
    """Compute pooled pixel-AP for one class.

    `predict_fn(img_path, img_pil)` must return a 2D map; the mask is resized
    to match its (H, W) before pooling.
    """
    items = _list_labeled_anomalies(dataset_dir, class_name)
    if not items:
        return float("nan")

    pred_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    it = tqdm(items, desc=f"  eval {class_name}", leave=False) if progress else items
    for it_d in it:
        img_pil = Image.open(it_d["img"]).convert("RGB")
        amap = predict_fn(it_d["img"], img_pil)
        amap = np.asarray(amap, dtype=np.float32)
        if amap.ndim != 2:
            raise ValueError(f"predict_fn returned shape {amap.shape}; expected (H, W)")
        m_pil = Image.open(it_d["mask"]).convert("L")
        m_pil = m_pil.resize((amap.shape[1], amap.shape[0]), Image.NEAREST)
        mask = (np.array(m_pil) > 0).astype(np.uint8)
        pred_chunks.append(amap.ravel())
        mask_chunks.append(mask.ravel())

    y_true = np.concatenate(mask_chunks)
    y_score = np.concatenate(pred_chunks)
    if y_true.sum() == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No positive class found in y_true")
        return float(average_precision_score(y_true, y_score))


def evaluate_all_classes(
    predict_fn_factory: Callable[[str], PredictFn],
    dataset_dir: Path,
    classes: List[str] | None = None,
) -> Dict[str, float]:
    """Run pooled per-class pixel-AP for every class.

    `predict_fn_factory(class_name) -> predict_fn` lets the caller load
    class-specific weights (e.g. one U-Net checkpoint per class) lazily.
    """
    dataset_dir = Path(dataset_dir)
    if classes is None:
        classes = sorted(
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith("class_")
        )
    results: Dict[str, float] = {}
    for cls in classes:
        predict_fn = predict_fn_factory(cls)
        ap = pooled_pixel_ap_for_class(predict_fn, cls, dataset_dir)
        results[cls] = ap
        print(f"  {cls}: pixel-AP = {ap:.4f}")
    mean = float(np.nanmean(list(results.values()))) if results else float("nan")
    results["mean"] = mean
    print(f"  MEAN: {mean:.4f}")
    return results


def save_results(results: Dict[str, float], out_path: Path, method_name: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": method_name, "per_class": results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_path}")
