import csv
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchvision
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.resnet import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
)

# ── Dataset scanning & anomaly description parsing ────────────────────────────

_VIEW_RE = re.compile(r'^(?P<stem>.+)_view(?P<view>[1-5])\.png$')

@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path:  str | None
    class_name: str
    is_anomaly: bool
    view_id:    int
    sample_id:  str

def _parse_view_and_id(filename: str) -> tuple[str, int]:
    m = _VIEW_RE.match(filename)
    if not m:
        raise ValueError(f'Unexpected filename: {filename}')
    return m.group('stem'), int(m.group('view')) - 1

def load_anomaly_descriptions(csv_path: str) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cls = row['public_class'].strip()
            rows.setdefault(cls, []).append(row)
    return rows

def load_pose_clusters(json_path: str) -> dict[str, dict[str, int]]:
    """Return {class_name: {image_basename: pose_label}}.

    The JSON only labels good train images (+ test). Anomalous train images
    are assigned downstream via NN to a labeled good of the same class.
    """
    with open(json_path) as f:
        raw = json.load(f)
    out: dict[str, dict[str, int]] = {}
    for cls, v in raw.items():
        d: dict[str, int] = {}
        for p, lbl in zip(v.get('train_paths', []), v.get('train_labels', [])):
            d[Path(p).name] = int(lbl)
        for p, lbl in zip(v.get('test_paths', []), v.get('test_labels', [])):
            d[Path(p).name] = int(lbl)
        out[cls] = d
    return out

def _downsampled_grayscale_features(samples, image_size: int = 32) -> np.ndarray:
    feats = []
    for s in samples:
        img = Image.open(s.image_path).convert('L').resize(
            (image_size, image_size), Image.BILINEAR)
        feats.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
    return (np.stack(feats) if feats
            else np.zeros((0, image_size * image_size), dtype=np.float32))

def motifs_from_description(text: str) -> list[str]:
    t = (text or '').lower()
    motifs: list[str] = []
    def add(name, *keys):
        if any(k in t for k in keys):
            motifs.append(name)
    add('scratch',   'scratch', 'groove', 'linear')
    add('stain',     'stain', 'blotchy', 'discolor', 'contamin')
    add('indent',    'depress', 'hollow', 'dimple', 'concave', 'dent')
    add('protrusion','raised', 'bulging', 'protrusion', 'extrusion')
    add('crack',     'crack', 'fissur', 'fract')
    add('fragment',  'fragment', 'broken', 'pieces', 'jagged')
    add('hole',      'hole', 'pitted')
    add('fuzzy',     'fuzzy', 'powder')
    if 'localized visual anomaly' in t or not motifs:
        motifs.append('localized')
    seen: set[str] = set()
    out: list[str] = []
    for m in motifs:
        if m not in seen:
            out.append(m); seen.add(m)
    return out

def build_motifs_for_class(desc_rows: list[dict] | None) -> list[str]:
    if not desc_rows:
        return ['scratch', 'stain', 'localized']
    allowed = {'scratch','stain','indent','protrusion','crack','fragment','hole','fuzzy','localized'}
    out: list[str] = []
    for r in desc_rows:
        for m in motifs_from_description(r.get('description', '')):
            if m in allowed and m not in out:
                out.append(m)
    return out or ['scratch', 'stain', 'localized']

def scan_class_dataset(dataset_dir: str, class_name: str) -> list[Sample]:
    base      = Path(dataset_dir) / class_name
    train_dir = base / 'train'
    gt_base   = base / 'ground_truth_train'
    samples: list[Sample] = []

    good_dir = train_dir / 'good'
    if not good_dir.exists():
        # Handle case where dataset structure might differ slightly (e.g. no train/good)
        return []

    for fn in sorted(os.listdir(good_dir)):
        if not fn.endswith('.png'):
            continue
        stem, view_id = _parse_view_and_id(fn)
        samples.append(Sample(str(good_dir / fn), None, class_name, False, view_id, stem))

    for sub in sorted(os.listdir(train_dir)):
        if not sub.startswith('anomaly_'):
            continue
        img_dir  = train_dir / sub
        mask_dir = gt_base   / sub
        if not img_dir.exists() or not mask_dir.exists():
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith('.png'):
                continue
            stem, view_id = _parse_view_and_id(fn)
            mask_path = mask_dir / fn
            if not mask_path.exists():
                raise FileNotFoundError(f'Missing mask: {mask_path}')
            samples.append(Sample(str(img_dir / fn), str(mask_path), class_name, True, view_id, stem))

    return samples

def split_by_sample_id(samples: list[Sample], val_ratio: float = 0.15,
                        seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    ids = sorted({s.sample_id for s in samples})
    rng.shuffle(ids)
    val_ids = set(ids[:max(1, int(len(ids) * val_ratio))])
    return [s for s in samples if s.sample_id not in val_ids], [s for s in samples if s.sample_id in val_ids]

# ── Anomaly Synthesis & CutPaste ──────────────────────────────────────────────

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr.transpose(2, 0, 1) / 255.0)

def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = (np.asarray(mask.convert('L'), dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    return torch.from_numpy(arr[None, ...])

def foreground_from_image(img: Image.Image, thr: int = 15) -> torch.Tensor:
    g  = np.asarray(img.convert('L'), dtype=np.uint8)
    fg = (g > thr).astype(np.float32)
    return torch.from_numpy(fg[None, ...])

def _fractal_noise_2d(h: int, w: int, octaves: int = 4, base_scale: float = 0.15,
                      rng: random.Random | None = None) -> np.ndarray:
    if rng is None:
        rng = random.Random()
    noise = np.zeros((h, w), dtype=np.float32)
    amp, freq = 1.0, base_scale
    for _ in range(octaves):
        gw = max(2, int(w * freq) + 2)
        gh = max(2, int(h * freq) + 2)
        grid = np.array([rng.gauss(0.0, 1.0) for _ in range(gw * gh)],
                        dtype=np.float32).reshape(gh, gw)
        xs = np.linspace(0, gw - 1, w)
        ys = np.linspace(0, gh - 1, h)
        x0 = np.floor(xs).astype(int).clip(0, gw - 2)
        y0 = np.floor(ys).astype(int).clip(0, gh - 2)
        x1 = (x0 + 1).clip(0, gw - 1)
        y1 = (y0 + 1).clip(0, gh - 1)
        fx = (xs - x0)[None, :]
        fy = (ys - y0)[:, None]
        layer = (  grid[np.ix_(y0, x0)] * (1 - fy) * (1 - fx)
                 + grid[np.ix_(y0, x1)] * (1 - fy) *      fx
                 + grid[np.ix_(y1, x0)] *      fy  * (1 - fx)
                 + grid[np.ix_(y1, x1)] *      fy  *      fx  )
        noise += amp * layer
        amp   *= 0.5
        freq  *= 2.0
    mn, mx = noise.min(), noise.max()
    return (noise - mn) / (mx - mn + 1e-8)

class AnomalyCropBank:
    MIN_PIXELS = 50
    def __init__(self, samples: list, image_size: int,
                 pose_lookup: dict[str, int] | None = None):
        """If pose_lookup is given (mapping good-image basename -> pose label),
        each crop is tagged with a pose label. Anomalous-train images that
        don't appear in pose_lookup are assigned to the nearest good sample's
        pose via 32x32 grayscale L2 NN (cheap, no model needed).
        """
        self.crops: list[tuple[np.ndarray, np.ndarray, int]] = []
        anomaly_samples = [s for s in samples
                           if s.is_anomaly and s.mask_path is not None]

        # Resolve a pose label per anomaly image.
        pose_for: dict[str, int] = {}
        if pose_lookup:
            good_with_pose = [
                (s, pose_lookup[Path(s.image_path).name])
                for s in samples
                if (not s.is_anomaly)
                and (Path(s.image_path).name in pose_lookup)
            ]
            for s in anomaly_samples:
                bn = Path(s.image_path).name
                if bn in pose_lookup:
                    pose_for[bn] = pose_lookup[bn]
            need_nn = [s for s in anomaly_samples
                       if Path(s.image_path).name not in pose_for]
            if good_with_pose and need_nn:
                good_feats  = _downsampled_grayscale_features(
                    [s for s, _ in good_with_pose])
                good_labels = np.asarray([lbl for _, lbl in good_with_pose])
                anom_feats  = _downsampled_grayscale_features(need_nn)
                for i, s in enumerate(need_nn):
                    d = np.linalg.norm(good_feats - anom_feats[i], axis=1)
                    pose_for[Path(s.image_path).name] = int(good_labels[d.argmin()])

        for s in anomaly_samples:
            img  = Image.open(s.image_path).convert('RGB').resize(
                       (image_size, image_size), Image.BILINEAR)
            mask = Image.open(s.mask_path).convert('L').resize(
                       (image_size, image_size), Image.NEAREST)
            img_np  = np.asarray(img,  dtype=np.float32)
            mask_np = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            if mask_np.sum() >= self.MIN_PIXELS:
                pose = pose_for.get(Path(s.image_path).name, -1)
                self.crops.append((img_np, mask_np, pose))

        self.by_pose: dict[int, list[int]] = {}
        for i, (_, _, p) in enumerate(self.crops):
            self.by_pose.setdefault(p, []).append(i)
        print(f'  AnomalyCropBank: {len(self.crops)} crops, '
              f'pose buckets={ {k: len(v) for k, v in self.by_pose.items()} }')

    def __len__(self) -> int:
        return len(self.crops)

    def sample(self, rng: random.Random,
               pose: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.crops:
            return None
        idxs = self.by_pose.get(pose) if pose is not None else None
        if not idxs:
            idxs = list(range(len(self.crops)))
        img, mask, _ = self.crops[rng.choice(idxs)]
        return img, mask

# ── Procedural shape generators ───────────────────────────────────────────────

def _generate_perlin_blob_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Fractal-noise blob in a random sub-window. Stain/contamination-like."""
    # Sub-window size 8–28 % of the long side. Keeps defects LOCAL.
    sh = rng.randint(max(8, h // 12), max(12, h // 4))
    sw = rng.randint(max(8, w // 12), max(12, w // 4))
    noise = _fractal_noise_2d(sh, sw, octaves=rng.randint(3, 5),
                              base_scale=rng.uniform(0.10, 0.30), rng=rng)
    thr = rng.uniform(0.45, 0.65)
    sub = (noise > thr).astype(np.float32)
    if rng.random() < 0.5:
        sigma = rng.uniform(0.5, 1.5)
        sub_pil = Image.fromarray((sub * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(sigma))
        sub = (np.asarray(sub_pil, dtype=np.float32) / 255.0 > 0.4).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    y0 = rng.randint(0, h - sh)
    x0 = rng.randint(0, w - sw)
    mask[y0:y0 + sh, x0:x0 + sw] = sub
    return mask

def _generate_bezier_stroke_mask(h: int, w: int,
                                 rng: random.Random) -> np.ndarray:
    """Quadratic / cubic Bezier rasterised by stamping discs along the curve.
    Best for scratch / groove / crack-like defects."""
    n_ctrl = rng.choice([3, 4])
    pad = max(2, min(h, w) // 20)
    pts = [(rng.randint(pad, w - 1 - pad), rng.randint(pad, h - 1 - pad))
           for _ in range(n_ctrl)]
    n_samples = max(w, h) * 3
    ts = np.linspace(0.0, 1.0, n_samples)
    P = np.asarray(pts, dtype=np.float64)
    if n_ctrl == 3:
        B = ((1 - ts)**2)[:, None] * P[0] \
          + (2 * (1 - ts) * ts)[:, None] * P[1] \
          + (ts**2)[:, None] * P[2]
    else:
        B = ((1 - ts)**3)[:, None] * P[0] \
          + (3 * (1 - ts)**2 * ts)[:, None] * P[1] \
          + (3 * (1 - ts) * ts**2)[:, None] * P[2] \
          + (ts**3)[:, None] * P[3]
    base_r = rng.randint(1, max(2, min(h, w) // 60))
    mask_pil = Image.new('L', (w, h), 0)
    draw     = ImageDraw.Draw(mask_pil)
    for i, (px, py) in enumerate(B):
        # subtle radius modulation along arc length
        r = base_r + (1 if rng.random() < 0.15 else 0) - (1 if rng.random() < 0.15 else 0)
        r = max(1, r)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=255)
    return np.asarray(mask_pil, dtype=np.float32) / 255.0

def _generate_random_walk_mask(h: int, w: int,
                               rng: random.Random) -> np.ndarray:
    """Branching random walk with momentum. Crack / fracture-like defects."""
    mask = np.zeros((h, w), dtype=np.float32)
    n_heads = rng.randint(1, 3)
    heads = [(rng.uniform(h * 0.15, h * 0.85),
              rng.uniform(w * 0.15, w * 0.85),
              rng.uniform(0.0, 2 * np.pi)) for _ in range(n_heads)]
    n_steps = rng.randint(80, 240)
    radius  = rng.randint(1, 2)
    for _ in range(n_steps):
        new_heads = []
        for y, x, theta in heads:
            theta = theta + rng.gauss(0.0, 0.35)
            step  = rng.uniform(1.0, 2.5)
            y    += step * np.sin(theta)
            x    += step * np.cos(theta)
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < h and 0 <= xi < w:
                y_lo, y_hi = max(0, yi - radius), min(h, yi + radius + 1)
                x_lo, x_hi = max(0, xi - radius), min(w, xi + radius + 1)
                mask[y_lo:y_hi, x_lo:x_hi] = 1.0
                new_heads.append((y, x, theta))
                # rare branching
                if rng.random() < 0.015 and len(heads) + len(new_heads) < 5:
                    new_heads.append((y, x, theta + rng.uniform(-1.2, 1.2)))
        heads = new_heads
        if not heads:
            break
    return mask

def _elastic_warp_mask(mask: np.ndarray, rng: random.Random,
                       max_disp: int = 18, grid: int = 5) -> np.ndarray:
    """Low-frequency displacement field warp. Same mask, wobbled boundary."""
    h, w = mask.shape
    dy_grid = np.array([rng.uniform(-max_disp, max_disp)
                        for _ in range(grid * grid)], dtype=np.float32
                       ).reshape(grid, grid)
    dx_grid = np.array([rng.uniform(-max_disp, max_disp)
                        for _ in range(grid * grid)], dtype=np.float32
                       ).reshape(grid, grid)
    dy = np.asarray(Image.fromarray(dy_grid, mode='F').resize((w, h), Image.BILINEAR),
                    dtype=np.float32)
    dx = np.asarray(Image.fromarray(dx_grid, mode='F').resize((w, h), Image.BILINEAR),
                    dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sy = np.clip(ys + dy, 0, h - 1).astype(np.int32)
    sx = np.clip(xs + dx, 0, w - 1).astype(np.int32)
    return mask[sy, sx]

def _cutmix_masks(a: np.ndarray, b: np.ndarray,
                  rng: random.Random) -> np.ndarray:
    """Combine two masks by spatial cut / union / intersection."""
    h, w = a.shape
    mode = rng.choices(['v', 'h', 'union', 'intersect'], weights=[3, 3, 2, 1])[0]
    if mode == 'v':
        cut = rng.randint(w // 4, 3 * w // 4)
        out = a.copy(); out[:, cut:] = b[:, cut:]
    elif mode == 'h':
        cut = rng.randint(h // 4, 3 * h // 4)
        out = a.copy(); out[cut:, :] = b[cut:, :]
    elif mode == 'union':
        out = np.maximum(a, b)
    else:
        out = a * b
        if out.sum() < AnomalyCropBank.MIN_PIXELS:
            out = np.maximum(a, b)
    return out


# ── Shape + texture bank (decoupled) ──────────────────────────────────────────

class ShapeTextureBank:
    """Drop-in replacement for AnomalyCropBank that decouples defect shape
    from defect texture.

    On each `sample()` it:
      1. Picks a SHAPE from {real mask, elastic-warped real, cutmix of two reals,
         Perlin blob, Bezier stroke, random walk} according to `shape_weights`.
      2. Picks a TEXTURE (bbox-cropped real defect pixels) independently.
      3. Composites the texture into the shape's bbox and returns the resulting
         (256x256 image, 256x256 mask) pair.

    The model only ever sees the SHAPE as its supervision target — guarantees
    every pixel inside the mask is "defective".
    """

    MIN_PIXELS = 50
    SHAPE_MODES = ('real', 'real_warp', 'real_cutmix',
                   'perlin', 'bezier', 'walk')

    def __init__(self, samples: list, image_size: int,
                 pose_lookup: dict[str, int] | None = None,
                 shape_weights: tuple[float, ...] = (0.30, 0.25, 0.10, 0.15, 0.15, 0.05)):
        assert len(shape_weights) == len(self.SHAPE_MODES)
        self.image_size    = image_size
        self.shape_weights = np.asarray(shape_weights, dtype=np.float64)
        self.shape_weights = self.shape_weights / self.shape_weights.sum()

        anomaly_samples = [s for s in samples
                           if s.is_anomaly and s.mask_path is not None]
        # Pose resolution (same logic as AnomalyCropBank)
        pose_for: dict[str, int] = {}
        if pose_lookup:
            good_with_pose = [
                (s, pose_lookup[Path(s.image_path).name])
                for s in samples
                if (not s.is_anomaly)
                and (Path(s.image_path).name in pose_lookup)
            ]
            for s in anomaly_samples:
                bn = Path(s.image_path).name
                if bn in pose_lookup:
                    pose_for[bn] = pose_lookup[bn]
            need_nn = [s for s in anomaly_samples
                       if Path(s.image_path).name not in pose_for]
            if good_with_pose and need_nn:
                good_feats  = _downsampled_grayscale_features(
                    [s for s, _ in good_with_pose])
                good_labels = np.asarray([lbl for _, lbl in good_with_pose])
                anom_feats  = _downsampled_grayscale_features(need_nn)
                for i, s in enumerate(need_nn):
                    d = np.linalg.norm(good_feats - anom_feats[i], axis=1)
                    pose_for[Path(s.image_path).name] = int(good_labels[d.argmin()])

        # Real defects → masks (full-frame) + textures (bbox-cropped)
        self.real_masks: list[tuple[np.ndarray, int]] = []
        self.textures:   list[tuple[np.ndarray, np.ndarray, int]] = []
        for s in anomaly_samples:
            img  = Image.open(s.image_path).convert('RGB').resize(
                       (image_size, image_size), Image.BILINEAR)
            mask = Image.open(s.mask_path).convert('L').resize(
                       (image_size, image_size), Image.NEAREST)
            img_np  = np.asarray(img,  dtype=np.float32)
            mask_np = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            if mask_np.sum() < self.MIN_PIXELS:
                continue
            pose = pose_for.get(Path(s.image_path).name, -1)
            self.real_masks.append((mask_np, pose))
            ys, xs = np.where(mask_np > 0.5)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            self.textures.append((img_np[y0:y1, x0:x1].copy(),
                                  mask_np[y0:y1, x0:x1].copy(),
                                  pose))

        self.masks_by_pose: dict[int, list[int]]    = {}
        self.textures_by_pose: dict[int, list[int]] = {}
        for i, (_, p) in enumerate(self.real_masks):
            self.masks_by_pose.setdefault(p, []).append(i)
        for i, (_, _, p) in enumerate(self.textures):
            self.textures_by_pose.setdefault(p, []).append(i)

        print(f'  ShapeTextureBank: {len(self.real_masks)} real shapes/textures, '
              f'pose buckets={ {k: len(v) for k, v in self.masks_by_pose.items()} }, '
              f'shape modes weighted as { {m: float(w) for m, w in zip(self.SHAPE_MODES, self.shape_weights)} }')

    def __len__(self) -> int:
        # Treat the bank as non-empty as long as we have ≥1 texture; shapes
        # can always be procedural.
        return len(self.textures)

    # ── shape sampling ────────────────────────────────────────────────────
    def _pick_real_mask(self, rng: random.Random,
                        pose: int | None) -> np.ndarray | None:
        if not self.real_masks:
            return None
        idxs = self.masks_by_pose.get(pose) if pose is not None else None
        if not idxs:
            idxs = list(range(len(self.real_masks)))
        return self.real_masks[rng.choice(idxs)][0].copy()

    def _sample_shape(self, rng: random.Random,
                      pose: int | None) -> np.ndarray | None:
        weights = self.shape_weights.copy()
        if not self.real_masks:  # can't do real / warp / cutmix without any
            weights[:3] = 0.0
            s = weights.sum()
            if s == 0:
                return None
            weights = weights / s
        mode = rng.choices(self.SHAPE_MODES, weights=weights.tolist(), k=1)[0]

        h = w = self.image_size
        if mode == 'real':
            return self._pick_real_mask(rng, pose)
        if mode == 'real_warp':
            m = self._pick_real_mask(rng, pose)
            return _elastic_warp_mask(m, rng) if m is not None else None
        if mode == 'real_cutmix':
            a = self._pick_real_mask(rng, pose)
            b = self._pick_real_mask(rng, None)  # any pose
            if a is None or b is None:
                return a if a is not None else b
            return _cutmix_masks(a, b, rng)
        if mode == 'perlin':
            return _generate_perlin_blob_mask(h, w, rng)
        if mode == 'bezier':
            return _generate_bezier_stroke_mask(h, w, rng)
        if mode == 'walk':
            return _generate_random_walk_mask(h, w, rng)
        return None

    # ── texture sampling ──────────────────────────────────────────────────
    def _sample_texture(self, rng: random.Random,
                        pose: int | None) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.textures:
            return None
        idxs = self.textures_by_pose.get(pose) if pose is not None else None
        if not idxs:
            idxs = list(range(len(self.textures)))
        rgb_bbox, alpha_bbox, _ = self.textures[rng.choice(idxs)]
        return rgb_bbox, alpha_bbox

    # ── composition ───────────────────────────────────────────────────────
    def _composite(self, shape: np.ndarray, rgb_bbox: np.ndarray,
                   alpha_bbox: np.ndarray,
                   rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
        ys, xs = np.where(shape > 0.5)
        if len(ys) == 0 or shape.sum() < self.MIN_PIXELS:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bh, bw = y1 - y0, x1 - x0
        if bh < 2 or bw < 2:
            return None

        # Resize the texture's RGB to fit the shape bbox. Stretch is fine for
        # blobs but ugly for thin shapes — for very elongated shapes we tile
        # the texture along the long axis instead.
        rgb_h, rgb_w = rgb_bbox.shape[:2]
        aspect_shape = bw / max(1, bh)
        aspect_tex   = rgb_w / max(1, rgb_h)
        if max(aspect_shape, 1 / aspect_shape) > 3.0 \
           and max(aspect_tex, 1 / aspect_tex) < 2.0:
            # Tile (avoids absurd stretching)
            tex_pil = Image.fromarray(np.clip(rgb_bbox, 0, 255).astype(np.uint8))
            # Resize source so the short side covers the shape's short side
            short_target = min(bh, bw)
            scale = short_target / max(1, min(rgb_h, rgb_w))
            tw = max(2, int(round(rgb_w * scale)))
            th = max(2, int(round(rgb_h * scale)))
            tex_pil = tex_pil.resize((tw, th), Image.BILINEAR)
            tile = np.asarray(tex_pil, dtype=np.float32)
            n_y = (bh + th - 1) // th
            n_x = (bw + tw - 1) // tw
            rgb_filled = np.tile(tile, (n_y, n_x, 1))[:bh, :bw]
        else:
            rgb_pil = Image.fromarray(np.clip(rgb_bbox, 0, 255).astype(np.uint8)).resize(
                (bw, bh), Image.BILINEAR)
            rgb_filled = np.asarray(rgb_pil, dtype=np.float32)

        out_img  = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        out_img[y0:y1, x0:x1] = rgb_filled
        return out_img, shape.astype(np.float32)

    # ── public sample() — same signature as AnomalyCropBank ───────────────
    MAX_COVERAGE = 0.30  # reject shapes covering more than 30 % of the frame

    def sample(self, rng: random.Random,
               pose: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        area = self.image_size * self.image_size
        for _ in range(6):  # retry a few times if a composition fails
            shape = self._sample_shape(rng, pose)
            if shape is None or shape.sum() < self.MIN_PIXELS:
                continue
            if shape.sum() > self.MAX_COVERAGE * area:
                continue
            tex = self._sample_texture(rng, pose)
            if tex is None:
                continue
            out = self._composite(shape, tex[0], tex[1], rng)
            if out is not None:
                return out
        return None


def _augment_defect_pair(src_img: np.ndarray, src_mask: np.ndarray,
                         rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Geometric + photometric augmentation of a (defect image, mask) pair.

    Applies random H/V flip, free rotation, mask dilation/erosion, an
    optional 'mirror-half' fold (reflects left half over right to invent
    new symmetric shapes from a single source), and brightness/contrast
    jitter on the image. Mask stays binary.
    """
    src_pil  = Image.fromarray(np.clip(src_img, 0, 255).astype(np.uint8))
    mask_pil = Image.fromarray((src_mask * 255).astype(np.uint8))

    if rng.random() < 0.5:
        src_pil  = src_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        src_pil  = src_pil.transpose(Image.FLIP_TOP_BOTTOM)
        mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-180.0, 180.0)
    src_pil  = src_pil.rotate(angle, resample=Image.BILINEAR,
                              expand=False, fillcolor=(0, 0, 0))
    mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST,
                               expand=False, fillcolor=0)

    r = rng.randint(-3, 3)
    if r > 0:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(2 * r + 1))
    elif r < 0:
        mask_pil = mask_pil.filter(ImageFilter.MinFilter(2 * (-r) + 1))

    if rng.random() < 0.2:
        m_arr  = np.asarray(mask_pil, dtype=np.uint8)
        im_arr = np.asarray(src_pil,  dtype=np.uint8).copy()
        half_w = m_arr.shape[1] // 2
        m_left  = m_arr[:, :half_w]
        im_left = im_arr[:, :half_w]
        mirror_m  = m_left[:, ::-1]
        mirror_im = im_left[:, ::-1]
        right_slice_m  = m_arr[:, half_w:half_w + mirror_m.shape[1]]
        right_slice_im = im_arr[:, half_w:half_w + mirror_m.shape[1]]
        new_m = np.maximum(right_slice_m, mirror_m)
        on    = mirror_m > 127
        right_slice_im[on] = mirror_im[on]
        m_new = m_arr.copy()
        m_new[:, half_w:half_w + mirror_m.shape[1]] = new_m
        mask_pil = Image.fromarray(m_new)
        src_pil  = Image.fromarray(im_arr)

    arr  = np.asarray(src_pil, dtype=np.float32)
    arr  = arr * rng.uniform(0.85, 1.15)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr  = (arr - mean) * rng.uniform(0.85, 1.15) + mean
    arr  = np.clip(arr, 0, 255)

    mask_new = (np.asarray(mask_pil, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    return arr, mask_new


def apply_cutpaste(dst_img: np.ndarray, dst_mask: np.ndarray,
                   src_img: np.ndarray, src_mask: np.ndarray,
                   rng: random.Random,
                   dst_foreground: np.ndarray | None = None,
                   feather_radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cut-paste a defect crop onto dst_img with geometric & photometric jitter.

    `dst_foreground` (HxW, 0/1, optional): if given, paste positions are
    rejected when <50% of the mask falls inside the foreground. We retry
    up to 8 positions before giving up.
    `feather_radius`: Gaussian-blur radius applied to the alpha used for
    *blending only*. The supervision mask written to `dst_mask` stays binary.
    """
    h, w = dst_img.shape[:2]
    src_img, src_mask = _augment_defect_pair(src_img, src_mask, rng)

    scale = rng.uniform(0.6, 1.5)
    new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
    src_pil = Image.fromarray(np.clip(src_img, 0, 255).astype(np.uint8)).resize(
                  (new_w, new_h), Image.BILINEAR)
    smask_pil = Image.fromarray((src_mask * 255).astype(np.uint8)).resize(
                  (new_w, new_h), Image.NEAREST)
    src_np  = np.asarray(src_pil,   dtype=np.float32)
    smask   = (np.asarray(smask_pil, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

    best = None
    for _ in range(8):
        dy = rng.randint(-new_h // 4, h - new_h // 4)
        dx = rng.randint(-new_w // 4, w - new_w // 4)
        sy1, sy2 = max(0, -dy), min(new_h, h - dy)
        sx1, sx2 = max(0, -dx), min(new_w, w - dx)
        dy1, dy2 = max(0,  dy), min(h, dy + new_h)
        dx1, dx2 = max(0,  dx), min(w, dx + new_w)
        if sy2 <= sy1 or sx2 <= sx1:
            continue
        pmask = smask[sy1:sy2, sx1:sx2]
        if pmask.sum() < 5:
            continue
        if dst_foreground is not None:
            fg_patch  = dst_foreground[dy1:dy2, dx1:dx2]
            on_fg     = (pmask * fg_patch).sum()
            if on_fg < 0.5 * pmask.sum():
                continue
        best = (dy1, dy2, dx1, dx2, sy1, sy2, sx1, sx2, pmask)
        break

    if best is None:
        return dst_img, dst_mask
    dy1, dy2, dx1, dx2, sy1, sy2, sx1, sx2, pmask = best
    patch = src_np[sy1:sy2, sx1:sx2]

    if feather_radius > 0:
        alpha_pil = Image.fromarray((pmask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(feather_radius))
        alpha = np.asarray(alpha_pil, dtype=np.float32) / 255.0
    else:
        alpha = pmask
    a = alpha[..., None]

    result_img  = dst_img.copy()
    result_mask = dst_mask.copy()
    result_img[dy1:dy2, dx1:dx2]  = result_img[dy1:dy2, dx1:dx2] * (1 - a) + patch * a
    result_mask[dy1:dy2, dx1:dx2] = np.maximum(result_mask[dy1:dy2, dx1:dx2], pmask)
    return result_img, result_mask

def _draw_random_blob(draw: ImageDraw.ImageDraw, w: int, h: int,
                      rng: random.Random) -> None:
    for _ in range(rng.randint(3, 10)):
        rx = rng.randint(max(2, w // 30), max(4, w // 10))
        ry = rng.randint(max(2, h // 30), max(4, h // 10))
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)

def synthesize_anomaly(img: Image.Image, motif: str,
                       rng: random.Random) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    mask_arr = np.zeros((h, w), dtype=np.float32)
    if motif == 'scratch':
        mask_pil = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_pil)
        for _ in range(rng.randint(1, 4)):
            x0, y0 = rng.randint(0, w-1), rng.randint(0, h-1)
            x1, y1 = rng.randint(0, w-1), rng.randint(0, h-1)
            draw.line((x0, y0, x1, y1), fill=255,
                      width=rng.randint(1, max(2, min(w, h) // 150)))
        mask_arr = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(1.0)),
                              dtype=np.float32) / 255.0
    elif motif in {'stain', 'fuzzy', 'localized'}:
        noise    = _fractal_noise_2d(h, w, octaves=4,
                                     base_scale=rng.uniform(0.05, 0.20), rng=rng)
        thr      = rng.uniform(0.45, 0.70)
        sigma    = 3.0 if motif == 'fuzzy' else 1.5
        mask_pil = Image.fromarray(((noise > thr).astype(np.uint8) * 255))
        mask_arr = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(sigma)),
                              dtype=np.float32) / 255.0
    elif motif in {'indent', 'protrusion'}:
        mask_pil = Image.new('L', (w, h), 0)
        _draw_random_blob(ImageDraw.Draw(mask_pil), w, h, rng)
        mask_arr = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(4.0)),
                              dtype=np.float32) / 255.0
    elif motif == 'crack':
        mask_pil = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_pil)
        pts = [(rng.randint(0, w-1), rng.randint(0, h-1))]
        for _ in range(rng.randint(3, 8)):
            x = int(np.clip(pts[-1][0] + rng.randint(-w//6, w//6), 0, w-1))
            y = int(np.clip(pts[-1][1] + rng.randint(-h//6, h//6), 0, h-1))
            pts.append((x, y))
        draw.line(pts, fill=255, width=rng.randint(1, 2))
        mask_arr = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(1.0)),
                              dtype=np.float32) / 255.0
    elif motif == 'fragment':
        mask_pil = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_pil)
        x0, y0 = rng.randint(0, w-1), rng.randint(0, h-1)
        bw, bh = rng.randint(w//20, w//6), rng.randint(h//20, h//6)
        draw.rectangle((x0, y0, min(w-1, x0+bw), min(h-1, y0+bh)), fill=255)
        mask_arr = np.asarray(mask_pil, dtype=np.float32) / 255.0
    elif motif == 'hole':
        mask_pil = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_pil)
        for _ in range(rng.randint(3, 20)):
            r  = rng.randint(1, max(2, min(w, h) // 120))
            cx, cy = rng.randint(0, w-1), rng.randint(0, h-1)
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)
        mask_arr = np.asarray(mask_pil, dtype=np.float32) / 255.0
    else:
        noise = _fractal_noise_2d(h, w, rng=rng)
        mask_arr = (noise > 0.6).astype(np.float32)

    mask_arr = np.clip(mask_arr, 0.0, 1.0)
    img_np   = np.asarray(img.convert('RGB'), dtype=np.float32)

    if motif in {'scratch', 'crack', 'hole', 'fragment'}:
        effect = img_np * (1.0 - rng.uniform(0.25, 0.65))
    elif motif in {'stain', 'localized'}:
        color  = np.array([rng.uniform(40, 180)] * 3, dtype=np.float32)
        alpha  = rng.uniform(0.25, 0.6)
        effect = (1 - alpha) * img_np + alpha * color
    elif motif == 'fuzzy':
        color  = np.array([rng.uniform(60,160), rng.uniform(120,220),
                           rng.uniform(60,160)], dtype=np.float32)
        alpha  = rng.uniform(0.25, 0.55)
        effect = (1 - alpha) * img_np + alpha * color
    else:
        effect = np.clip(img_np + 255.0 * rng.uniform(-0.25, 0.25), 0, 255)

    out = img_np * (1 - mask_arr[..., None]) + effect * mask_arr[..., None]
    return (Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)),
            Image.fromarray((mask_arr * 255).astype(np.uint8)))

# ── Dataset ───────────────────────────────────────────────────────────────────

class ADLSegmentationDataset(Dataset):
    def __init__(self, samples: list, image_size: int = 256, train: bool = True,
                 p_synth: float = 0.30, p_cutpaste: float = 0.20,
                 p_multi_paste: float = 0.30,
                 motifs: list[str] | None = None,
                 crop_bank: AnomalyCropBank | None = None,
                 pose_lookup: dict[str, int] | None = None,
                 seed: int = 0):
        self.samples       = samples
        self.image_size    = image_size
        self.train         = train
        self.p_synth       = p_synth
        self.p_cutpaste    = p_cutpaste
        self.p_multi_paste = p_multi_paste
        self.motifs        = motifs or ['scratch', 'stain', 'localized']
        self.crop_bank     = crop_bank
        self.pose_lookup   = pose_lookup or {}
        self.seed          = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s   = self.samples[idx]
        rng = random.Random(self.seed + idx)
        img  = Image.open(s.image_path).convert('RGB').resize(
                   (self.image_size, self.image_size), Image.BILINEAR)
        mask = (Image.open(s.mask_path).convert('L').resize(
                   (self.image_size, self.image_size), Image.NEAREST)
                if s.mask_path else Image.new('L', (self.image_size, self.image_size), 0))
        img_np  = np.asarray(img,  dtype=np.float32)
        mask_np = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        if self.train and not s.is_anomaly:
            roll = rng.random()
            if (self.crop_bank and len(self.crop_bank) > 0
                    and roll < self.p_cutpaste):
                pose   = self.pose_lookup.get(Path(s.image_path).name)
                fg_arr = (np.asarray(img.convert('L'), dtype=np.uint8) > 15).astype(np.float32)
                sampled = self.crop_bank.sample(rng, pose=pose)
                if sampled is not None:
                    src_img, src_mask = sampled
                    img_np, mask_np = apply_cutpaste(
                        img_np, mask_np, src_img, src_mask, rng,
                        dst_foreground=fg_arr)
                    if rng.random() < self.p_multi_paste:
                        sampled2 = self.crop_bank.sample(rng, pose=pose)
                        if sampled2 is not None:
                            img_np, mask_np = apply_cutpaste(
                                img_np, mask_np, sampled2[0], sampled2[1], rng,
                                dst_foreground=fg_arr)
            elif roll < self.p_cutpaste + self.p_synth:
                motif   = rng.choice(self.motifs)
                pil_tmp = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                pil_out, syn_pil = synthesize_anomaly(pil_tmp, motif, rng)
                img_np  = np.asarray(pil_out, dtype=np.float32)
                syn_np  = (np.asarray(syn_pil, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
                mask_np = np.maximum(mask_np, syn_np)
        if self.train and rng.random() < 0.5:
            pil_blur = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            pil_blur = pil_blur.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.8)))
            img_np   = np.asarray(pil_blur, dtype=np.float32)
        img_final  = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        mask_final = Image.fromarray((mask_np * 255).astype(np.uint8))
        return {
            'image':      pil_to_tensor(img_final),
            'mask':       mask_to_tensor(mask_final),
            'foreground': foreground_from_image(img_final),
            'view_id':    torch.tensor(s.view_id, dtype=torch.long),
            'is_anomaly': torch.tensor(1 if s.is_anomaly else 0, dtype=torch.long),
            'sample_id':  s.sample_id,
            'image_path': s.image_path,
        }

# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    encoder:            str  = 'resnet34'
    pretrained:         bool = True
    view_count:         int  = 5
    view_embed_dim:     int  = 64
    base_channels:      int  = 256
    use_attention_gates: bool = True

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_ch)
        self.to_beta  = nn.Linear(cond_dim, feat_ch)
    def forward(self, x, cond):
        g = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + g) + b

class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g  = nn.Conv2d(g_ch,    inter_ch, kernel_size=1, bias=True)
        self.W_x  = nn.Conv2d(x_ch,    inter_ch, kernel_size=1, bias=False)
        self.psi  = nn.Conv2d(inter_ch, 1,        kernel_size=1, bias=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(self.W_g(g), size=x.shape[-2:],
                             mode='bilinear', align_corners=False)
        att  = torch.sigmoid(self.psi(F.relu(g_up + self.W_x(x), inplace=True)))
        return x * att

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim,
                 use_attention: bool = True):
        super().__init__()
        self.attn = (
            AttentionGate(g_ch=in_ch, x_ch=skip_ch,
                          inter_ch=max(1, skip_ch // 2))
            if use_attention else None
        )
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)
        self.film  = FiLM(cond_dim, out_ch)

    def forward(self, x, skip, cond):
        if self.attn is not None:
            skip = self.attn(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.film(x, cond)
        x = self.conv2(x)
        return x

class ResNetEncoder(nn.Module):
    def __init__(self, name: str = 'resnet34', pretrained: bool = True):
        super().__init__()
        if name == 'resnet18':
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_channels = (64, 64, 128, 256, 512)
        elif name == 'resnet34':
            m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_channels = (64, 64, 128, 256, 512)
        elif name == 'resnet50':
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.out_channels = (64, 256, 512, 1024, 2048)
        else:
            raise ValueError(f'Unknown encoder: {name}')
        self.stem    = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.maxpool = m.maxpool
        self.layer1  = m.layer1
        self.layer2  = m.layer2
        self.layer3  = m.layer3
        self.layer4  = m.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

class MultiViewUNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg     = cfg
        self.encoder = ResNetEncoder(cfg.encoder, cfg.pretrained)
        enc0, enc1, enc2, enc3, enc4 = self.encoder.out_channels
        dec = cfg.base_channels
        self.view_embed = nn.Embedding(cfg.view_count, cfg.view_embed_dim)
        self.cond_mlp   = nn.Sequential(
            nn.Linear(cfg.view_embed_dim, cfg.view_embed_dim), nn.SiLU(inplace=True),
            nn.Linear(cfg.view_embed_dim, cfg.view_embed_dim),
        )
        cond_dim = cfg.view_embed_dim
        ua = cfg.use_attention_gates
        self.bottleneck = nn.Sequential(ConvBNAct(enc4, dec), ConvBNAct(dec, dec))
        self.up3 = UpBlock(dec,     enc3, dec//2, cond_dim, ua)
        self.up2 = UpBlock(dec//2,  enc2, dec//4, cond_dim, ua)
        self.up1 = UpBlock(dec//4,  enc1, dec//8, cond_dim, ua)
        self.up0 = UpBlock(dec//8,  enc0, dec//8, cond_dim, ua)
        self.head = nn.Sequential(
            ConvBNAct(dec//8, dec//16),
            nn.Conv2d(dec//16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 256 or x.shape[-2] != 256:
             x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        cond = self.cond_mlp(self.view_embed(view_id.long()))
        x0, x1, x2, x3, x4 = self.encoder(x)
        b  = self.bottleneck(x4)
        d3 = self.up3(b,  x3, cond)
        d2 = self.up2(d3, x2, cond)
        d1 = self.up1(d2, x1, cond)
        d0 = self.up0(d1, x0, cond)
        d0 = F.interpolate(d0, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.head(d0)

# ── Loss & metrics ────────────────────────────────────────────────────────────

def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits).view(logits.shape[0], -1)
    t = targets.view(targets.shape[0], -1)
    inter = (p * t).sum(1)
    return 1.0 - ((2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()

def bce_dice_loss(logits, targets, foreground_mask=None, bce_weight=0.5):
    if foreground_mask is not None:
        logits  = logits  * foreground_mask
        targets = targets * foreground_mask
    return (bce_weight * F.binary_cross_entropy_with_logits(logits, targets)
            + (1 - bce_weight) * dice_loss(logits, targets))

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float().view(logits.shape[0], -1)
    t     = targets.view(targets.shape[0], -1)
    return ((2 * (preds * t).sum(1) + eps) / (preds.sum(1) + t.sum(1) + eps)).mean()

# ── Training & Inference ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, dices = [], []
    for batch in loader:
        x, y, fg, vid = (batch[k].to(device) for k in
                         ('image', 'mask', 'foreground', 'view_id'))
        logits = model(x, vid)
        losses.append(bce_dice_loss(logits, y, fg).item())
        dices.append(dice_score_from_logits(logits * fg, y * fg).item())
    return {'loss': float(np.mean(losses)), 'dice': float(np.mean(dices))}

def train_one_class(
    dataset_dir, class_name, desc_csv, out_dir,
    image_size=256, batch_size=16, epochs=12, lr=2e-4,
    val_ratio=0.15, p_synth=0.4, p_cutpaste=0.3, p_multi_paste=0.3, seed=42,
    encoder='resnet34', use_attention_gates=True,
    pose_json: str | None = None,
    use_shape_bank: bool = True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    desc   = load_anomaly_descriptions(desc_csv)
    motifs = build_motifs_for_class(desc.get(class_name))
    print(f'  motifs: {motifs}')
    pose_lookup = None
    if pose_json:
        try:
            pose_lookup = load_pose_clusters(pose_json).get(class_name)
            print(f'  pose: {len(pose_lookup) if pose_lookup else 0} good images labeled')
        except FileNotFoundError:
            print(f'  pose: {pose_json} not found, skipping pose-aware sampling')
    samples         = scan_class_dataset(dataset_dir, class_name)
    train_s, val_s  = split_by_sample_id(samples, val_ratio, seed)
    BankCls         = ShapeTextureBank if use_shape_bank else AnomalyCropBank
    crop_bank       = BankCls(train_s, image_size, pose_lookup=pose_lookup)
    train_ds = ADLSegmentationDataset(
        train_s, image_size, train=True,
        p_synth=p_synth, p_cutpaste=p_cutpaste, p_multi_paste=p_multi_paste,
        motifs=motifs, crop_bank=crop_bank, pose_lookup=pose_lookup, seed=seed)
    val_ds = ADLSegmentationDataset(
        val_s, image_size, train=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    cfg   = ModelConfig(encoder=encoder, pretrained=True,
                        view_count=5, use_attention_gates=use_attention_gates)
    model = MultiViewUNet(cfg).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    outp = Path(out_dir) / class_name
    outp.mkdir(parents=True, exist_ok=True)
    best_path = str(outp / 'best.pt')
    best_val  = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'{class_name} {epoch}/{epochs}')
        for batch in pbar:
            x, y, fg, vid = (batch[k].to(device, non_blocking=True)
                             for k in ('image','mask','foreground','view_id'))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = bce_dice_loss(model(x, vid), y, fg)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        m = evaluate(model, val_loader, device)
        with (outp / 'metrics.csv').open('a') as f:
            f.write(f"{epoch},{m['loss']:.6f},{m['dice']:.6f}\n")
        if m['loss'] < best_val:
            best_val = m['loss']
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)
    return best_path

@torch.no_grad()
def load_model(ckpt_path: str, device: torch.device) -> MultiViewUNet:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg  = ModelConfig(**ckpt.get('cfg', {}))
    m    = MultiViewUNet(cfg)
    m.load_state_dict(ckpt['model'], strict=True)
    return m.to(device).eval()

@torch.no_grad()
def _infer_single_tta(model, img_rs: Image.Image, view_id_t: torch.Tensor,
                      device, use_tta: bool = True) -> np.ndarray:
    flips = ['none', 'h', 'v'] if use_tta else ['none']
    preds = []
    for flip in flips:
        if   flip == 'h': img_f = img_rs.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip == 'v': img_f = img_rs.transpose(Image.FLIP_TOP_BOTTOM)
        else:             img_f = img_rs
        x    = pil_to_tensor(img_f).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x, view_id_t))[0, 0].cpu().numpy()
        if   flip == 'h': prob = prob[:, ::-1].copy()
        elif flip == 'v': prob = prob[::-1, :].copy()
        preds.append(prob)
    return np.mean(preds, axis=0)

def _aggregate_views(probs: list[np.ndarray],
                     boost_threshold: float = 0.25) -> list[np.ndarray]:
    max_conf = max(float(p.max()) for p in probs)
    if max_conf <= boost_threshold:
        return probs
    factor = 1.0 + 0.5 * (max_conf - boost_threshold) / (1.0 - boost_threshold)
    return [np.clip(p * factor, 0.0, 1.0) for p in probs]

def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q    = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts   = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, flat.size]
    parts  = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)
