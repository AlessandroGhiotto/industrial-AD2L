"""Save/load raw anomaly maps for ensembling.

Each model's pipeline calls `dump_labeled()` and `dump_test()` to write
per-class .npz files under `<root>/<model_name>/<class_name>/`. The ensemble
script (`scripts/ensemble_weighted.py`) loads these via `load_labeled()` and
`load_test()` to fit per-class weights and produce a combined submission.

Maps are stored as float16 to keep disk footprint manageable (a 256x256 map is
128 KB at fp16). The precision drop is well below the q8rle quantization floor
applied at submission time.
"""
from pathlib import Path

import numpy as np


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def dump_labeled(out_root, model_name, class_name, results):
    """Save labeled-eval results for one class.

    results: list of dicts with keys
        - 'anomaly_map' (H, W) or array convertible to it
        - 'mask' (H, W) or (1, H, W)
        - 'label' int
        - 'path' str
    """
    out_dir = Path(out_root) / model_name / class_name
    _ensure_dir(out_dir)
    maps = np.stack([np.asarray(r['anomaly_map'], dtype=np.float16) for r in results], axis=0)
    masks = np.stack([np.asarray(r['mask']).squeeze().astype(np.uint8) for r in results], axis=0)
    paths = np.array([str(r['path']) for r in results])
    labels = np.array([int(r['label']) for r in results], dtype=np.uint8)
    np.savez_compressed(
        out_dir / "labeled.npz",
        maps=maps, masks=masks, paths=paths, labels=labels,
    )


def dump_test(out_root, model_name, class_name, results):
    """Save test-set predictions for one class.

    results: list of dicts with 'anomaly_map' and 'path'.
    """
    out_dir = Path(out_root) / model_name / class_name
    _ensure_dir(out_dir)
    maps = np.stack([np.asarray(r['anomaly_map'], dtype=np.float16) for r in results], axis=0)
    paths = np.array([str(r['path']) for r in results])
    np.savez_compressed(out_dir / "test.npz", maps=maps, paths=paths)


def dump_good_eval(out_root, model_name, class_name, results):
    """Save held-out good-image predictions for one class (label=0, mask=all zeros).

    Used by the ensemble weight-fitter to include true-negative pixels alongside
    the anomaly-labeled ones. Without this, AP on the labeled pool ignores false
    positives entirely, which is the dominant source of in-sample → Kaggle gap.

    results: list of dicts with 'anomaly_map' and 'path'.
    """
    out_dir = Path(out_root) / model_name / class_name
    _ensure_dir(out_dir)
    maps = np.stack([np.asarray(r['anomaly_map'], dtype=np.float16) for r in results], axis=0)
    paths = np.array([str(r['path']) for r in results])
    np.savez_compressed(out_dir / "good.npz", maps=maps, paths=paths)


def load_labeled(in_root, model_name, class_name):
    f = Path(in_root) / model_name / class_name / "labeled.npz"
    if not f.exists():
        raise FileNotFoundError(f"Missing labeled dump: {f}")
    z = np.load(f)
    return {
        "maps": z["maps"].astype(np.float32),
        "masks": z["masks"].astype(np.uint8),
        "paths": z["paths"].tolist(),
        "labels": z["labels"].astype(np.uint8),
    }


def load_test(in_root, model_name, class_name):
    f = Path(in_root) / model_name / class_name / "test.npz"
    if not f.exists():
        raise FileNotFoundError(f"Missing test dump: {f}")
    z = np.load(f)
    return {
        "maps": z["maps"].astype(np.float32),
        "paths": z["paths"].tolist(),
    }


def load_good_eval(in_root, model_name, class_name):
    """Load held-out good-image predictions. Returns None if the file does not exist
    (callers should treat missing dumps as 'this model has no good-eval data')."""
    f = Path(in_root) / model_name / class_name / "good.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {
        "maps": z["maps"].astype(np.float32),
        "paths": z["paths"].tolist(),
    }


def align_by_path(canonical_paths, paths, maps):
    """Reorder `maps` so its rows match `canonical_paths` (matched by Path.stem).

    Used to align model outputs that may have been produced in different orders.
    """
    from pathlib import Path as _P
    lookup = {_P(p).stem: i for i, p in enumerate(paths)}
    order = [lookup[_P(p).stem] for p in canonical_paths]
    return maps[order]
