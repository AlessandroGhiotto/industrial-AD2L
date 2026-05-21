import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms as T
from tqdm.auto import tqdm

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

from adl_lib.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
    SEED,
)

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ===== Utilities =====


def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    return torch.from_numpy(mask)


def find_matching_mask(image_path: Path, mask_dir: Path) -> Path:
    """
    Robustly find a mask file matching the image.
    Tries multiple naming conventions common in anomaly detection datasets.
    """
    image_path = Path(image_path)
    mask_dir = Path(mask_dir)

    if not mask_dir.exists():
        return None

    stem = image_path.stem
    candidates = []

    # Strategy 1: exact stem match with different extensions
    for ext in IMG_EXTENSIONS:
        candidates.append(mask_dir / f"{stem}{ext}")
        candidates.append(mask_dir / f"{stem}_mask{ext}")
        candidates.append(mask_dir / f"{stem}_gt{ext}")

    for c in candidates:
        if c.exists():
            return c

    # Strategy 2: prefix match (in case files share a sample_id prefix)
    all_masks = list_images(mask_dir)
    for m in all_masks:
        # Check if image stem is in mask stem or vice versa
        if stem in m.stem or m.stem in stem:
            return m
        # Also try matching by prefix (first N characters)
        if len(stem) >= 3 and len(m.stem) >= 3:
            if stem[:5] == m.stem[:5]:
                return m

    return None


def make_transforms(image_size: int):
    train_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_tf, eval_tf


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = load_rgb(path)
        x = self.transform(img)
        return {"image": x, "path": str(path)}


class PixelValDataset(Dataset):
    def __init__(self, items: List[dict], transform, image_size: int):
        self.items = items
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = (
                load_rgb(item["image_path"])
                if not isinstance(item["image_path"], torch.Tensor)
                else item["image_path"]
            )
            if img is None:
                raise ValueError(f"Failed to load image from {item['image_path']}")
            x = self.transform(img)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a blank image as fallback
            blank_img = Image.new(
                "RGB", (self.image_size, self.image_size), color=(0, 0, 0)
            )
            x = self.transform(blank_img)

        try:
            if item["mask_path"] is None:
                mask = torch.zeros(
                    (self.image_size, self.image_size), dtype=torch.uint8
                )
            else:
                mask = load_mask(Path(item["mask_path"]), self.image_size)
        except Exception as e:
            print(f"Error loading mask {item['mask_path']}: {e}")
            # Return a blank mask as fallback
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.uint8)

        return {
            "image": x,
            "mask": mask,
            "label": int(item["label"]),
            "path": str(item["image_path"]),
            "mask_path": "" if item["mask_path"] is None else str(item["mask_path"]),
            "anomaly_type": item.get("anomaly_type", "good"),
        }


def build_student_teacher_splits(
    path: Path,
    class_name: str,
    image_size: int,
    seed: int = 7,
):
    """
    Build splits for Student-Teacher training.

    Train: ALL good images from train/good/ (maximize normal data)
    Val:   Labeled anomalies from train/anomaly_YY/ with ground truth masks

    This is the correct setup: train only on normals, validate on scarce labeled anomalies.
    """
    rng = random.Random(seed)

    class_root = Path(path) / class_name

    good_dir = class_root / "train" / "good"
    good_paths = list_images(good_dir)

    if len(good_paths) == 0:
        raise RuntimeError(f"No good images found in: {good_dir}")

    # Use ALL good images for training (no hold-out split)
    train_good_paths = sorted(good_paths)

    # Validation: only labeled anomalies with masks
    val_items = []

    train_root = class_root / "train"
    gt_root = class_root / "ground_truth_train"

    anomaly_dirs = (
        sorted([d for d in train_root.iterdir() if d.is_dir() and d.name != "good"])
        if train_root.exists()
        else []
    )

    for anomaly_dir in anomaly_dirs:
        anomaly_type = anomaly_dir.name
        mask_dir = gt_root / anomaly_type

        anomaly_images = list_images(anomaly_dir)

        for img_path in anomaly_images:
            mask_path = find_matching_mask(img_path, mask_dir)

            if mask_path is None:
                print(f"Warning: no mask found for {img_path}")

            val_items.append(
                {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "anomaly_type": anomaly_type,
                }
            )

    return train_good_paths, val_items


def validate_student_teacher_splits(train_good_paths, val_items):
    """
    Validate and report dataset splits.
    Use this before training to ensure data is loaded correctly.
    """
    print("=" * 80)
    print("STUDENT-TEACHER DATASET VALIDATION")
    print("=" * 80)

    print(f"\n✓ Training images (good only): {len(train_good_paths)}")
    if len(train_good_paths) > 0:
        print(f"  Example: {train_good_paths[0]}")

    # Count validation items
    val_good_items = [item for item in val_items if item["label"] == 0]
    val_anom_items = [item for item in val_items if item["label"] == 1]
    val_anom_with_mask = [
        item for item in val_anom_items if item["mask_path"] is not None
    ]
    val_anom_without_mask = [
        item for item in val_anom_items if item["mask_path"] is None
    ]

    print(f"\n✓ Validation good images: {len(val_good_items)}")
    if len(val_good_items) > 0:
        print(f"  Example: {val_good_items[0]['image_path']}")
        print(f"  Mask path: {val_good_items[0]['mask_path']}")

    print(f"\n✓ Validation anomalies: {len(val_anom_items)}")
    print(f"  - With masks: {len(val_anom_with_mask)}")
    print(f"  - Without masks: {len(val_anom_without_mask)}")

    if len(val_anom_with_mask) > 0:
        print(f"\n  Example with mask:")
        item = val_anom_with_mask[0]
        print(f"    Image: {item['image_path']}")
        print(f"    Mask: {item['mask_path']}")
        print(f"    Type: {item['anomaly_type']}")

    if len(val_anom_without_mask) > 0:
        print(f"\n  ⚠️  Examples WITHOUT mask (will be treated as zero-mask):")
        for item in val_anom_without_mask[:3]:
            print(f"    Image: {item['image_path']}")
            print(f"    Type: {item['anomaly_type']}")
        if len(val_anom_without_mask) > 3:
            print(f"    ... and {len(val_anom_without_mask) - 3} more")

    print(f"\n" + "=" * 80)
    if len(val_anom_without_mask) > 0:
        print("⚠️  WARNING: Some anomalies have no ground truth masks!")
        print("    This will hurt pixel-level AP computation.")
        print("    Check mask directory paths and file naming conventions.")
    else:
        print("✓ All anomalies have ground truth masks!")
    print("=" * 80 + "\n")

    return {
        "n_train_good": len(train_good_paths),
        "n_val_good": len(val_good_items),
        "n_val_anom_with_mask": len(val_anom_with_mask),
        "n_val_anom_without_mask": len(val_anom_without_mask),
    }


# ===== Student-Teacher model =====


class StudentTeacherAD(nn.Module):
    def __init__(
        self,
        teacher_backbone: str = "wide_resnet50_2",
        student_backbone: str = None,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        n_students: int = 3,
        student_pretrained: bool = False,
        device: str = DEVICE,
    ):
        super().__init__()

        self.device = device
        self.n_students = int(n_students)
        self.score_stats = None

        tb = teacher_backbone
        sb = student_backbone if student_backbone is not None else teacher_backbone

        self.teacher = timm.create_model(
            tb,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )

        self.students = nn.ModuleList(
            [
                timm.create_model(
                    sb,
                    pretrained=student_pretrained,
                    features_only=True,
                    out_indices=out_indices,
                )
                for _ in range(self.n_students)
            ]
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher.eval()
        # Build per-student, per-layer 1x1 projectors to align student channels to teacher channels
        # Use a dummy forward pass to infer channel sizes.
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                t_feats = self.teacher(dummy)
                teacher_channels = [t.shape[1] for t in t_feats]

                student_projectors = []
                for student in self.students:
                    s_feats = student(dummy)
                    proj_layers = []
                    for s_ft, t_ch in zip(s_feats, teacher_channels):
                        s_ch = s_ft.shape[1]
                        if s_ch != t_ch:
                            proj_layers.append(nn.Conv2d(s_ch, t_ch, kernel_size=1))
                        else:
                            proj_layers.append(nn.Identity())
                    student_projectors.append(nn.ModuleList(proj_layers))

                self.student_projectors = nn.ModuleList(student_projectors)
        except Exception:
            # Fallback: if anything goes wrong, create identity projectors
            self.student_projectors = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Identity()
                            for _ in range(len(self.teacher.feature_info.channels))
                        ]
                    )
                    for _ in range(self.n_students)
                ]
            )

    def forward(self, x):
        with torch.no_grad():
            teacher_features = self.teacher(x)

        raw_students_features = [student(x) for student in self.students]

        # Apply projectors to align student channel dims to teacher channels
        students_features = []
        for proj_list, s_feats in zip(self.student_projectors, raw_students_features):
            mapped = [proj(sf) for proj, sf in zip(proj_list, s_feats)]
            students_features.append(mapped)

        return teacher_features, students_features

    def set_score_stats(self, score_stats: dict):
        self.score_stats = score_stats

    def has_score_stats(self) -> bool:
        return self.score_stats is not None


def student_teacher_loss(teacher_features, student_features):
    loss = 0.0
    for t, s in zip(teacher_features, student_features):
        t = F.normalize(t.detach(), p=2, dim=1)
        s = F.normalize(s, p=2, dim=1)
        loss = loss + torch.mean((s - t) ** 2)
    return loss / len(teacher_features)


@torch.no_grad()
def compute_student_teacher_components(
    model: StudentTeacherAD, images: torch.Tensor, image_size: int
):
    model.eval()

    teacher_features, students_features = model(images)

    err_layers = []
    var_layers = []

    for layer_idx, t in enumerate(teacher_features):
        t = F.normalize(t, p=2, dim=1)
        student_stack = torch.stack(
            [
                F.normalize(student_feats[layer_idx], p=2, dim=1)
                for student_feats in students_features
            ],
            dim=1,
        )

        mean_students = student_stack.mean(dim=1)
        err = ((mean_students - t) ** 2).mean(dim=1, keepdim=True)

        var = ((student_stack - mean_students.unsqueeze(1)) ** 2).mean(dim=1)
        var = var.mean(dim=1, keepdim=True)

        err = F.interpolate(
            err, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        var = F.interpolate(
            var, size=(image_size, image_size), mode="bilinear", align_corners=False
        )

        err_layers.append(err)
        var_layers.append(var)

    err_map = torch.mean(torch.stack(err_layers, dim=0), dim=0)[:, 0]
    var_map = torch.mean(torch.stack(var_layers, dim=0), dim=0)[:, 0]

    return err_map, var_map


@torch.no_grad()
def compute_student_teacher_maps(
    model: StudentTeacherAD, images: torch.Tensor, image_size: int
):
    err_map, var_map = compute_student_teacher_components(
        model=model, images=images, image_size=image_size
    )

    if model.has_score_stats():
        stats = model.score_stats
        err_map = (err_map - stats["err_mean"]) / stats["err_std"]
        var_map = (var_map - stats["var_mean"]) / stats["var_std"]

    return err_map + var_map


def image_score_from_map(anomaly_map: torch.Tensor, topk_percent: float = 1.0):
    b, h, w = anomaly_map.shape
    flat = anomaly_map.reshape(b, -1)

    k = max(1, int(flat.shape[1] * topk_percent / 100.0))
    topk_values = torch.topk(flat, k=k, dim=1).values

    return topk_values.mean(dim=1)


@torch.no_grad()
def calibrate_student_teacher_scores(
    model: StudentTeacherAD,
    loader: DataLoader,
    image_size: int,
    device: str = DEVICE,
    eps: float = 1e-6,
):
    model.eval()

    err_sum = 0.0
    err_sum_sq = 0.0
    err_count = 0

    var_sum = 0.0
    var_sum_sq = 0.0
    var_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        err_map, var_map = compute_student_teacher_components(
            model=model, images=images, image_size=image_size
        )

        err_flat = err_map.reshape(-1).float()
        var_flat = var_map.reshape(-1).float()

        err_sum += float(err_flat.sum().item())
        err_sum_sq += float((err_flat**2).sum().item())
        err_count += int(err_flat.numel())

        var_sum += float(var_flat.sum().item())
        var_sum_sq += float((var_flat**2).sum().item())
        var_count += int(var_flat.numel())

    err_mean = err_sum / max(err_count, 1)
    err_var = max(err_sum_sq / max(err_count, 1) - err_mean**2, eps)
    err_std = float(np.sqrt(err_var))

    var_mean = var_sum / max(var_count, 1)
    var_var = max(var_sum_sq / max(var_count, 1) - var_mean**2, eps)
    var_std = float(np.sqrt(var_var))

    model.set_score_stats(
        {
            "err_mean": err_mean,
            "err_std": err_std,
            "var_mean": var_mean,
            "var_std": var_std,
        }
    )

    return model.score_stats


# ===== Training & evaluation =====

ST_SAVE_DIR = Path("./student_teacher_models")
ST_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def train_student_teacher(
    path,
    class_name,
    image_size: int = 224,
    backbone: str = "wide_resnet50_2",
    student_backbone: str = None,
    out_indices: Tuple[int, ...] = (1, 2, 3),
    n_students: int = 3,
    student_pretrained: bool = False,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 7,
    device: str = DEVICE,
    topk_percent: float = 1.0,
):
    set_seed(seed)

    train_tf, eval_tf = make_transforms(image_size)

    train_good_paths, val_items = build_student_teacher_splits(
        path=path,
        class_name=class_name,
        image_size=image_size,
        seed=seed,
    )

    train_ds = ImagePathDataset(train_good_paths, transform=train_tf)
    calib_ds = ImagePathDataset(train_good_paths, transform=eval_tf)
    val_ds = PixelValDataset(val_items, transform=eval_tf, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    calib_loader = DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    model = StudentTeacherAD(
        teacher_backbone=backbone,
        student_backbone=student_backbone,
        out_indices=out_indices,
        n_students=n_students,
        student_pretrained=student_pretrained,
        device=device,
    ).to(device)

    optimizers = []
    for si in range(len(model.students)):
        params = list(model.students[si].parameters())
        if hasattr(model, "student_projectors"):
            params += list(model.student_projectors[si].parameters())
        optimizers.append(torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay))

    best_pixel_ap = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.teacher.eval()
        for student in model.students:
            student.train()

        train_losses = [[] for _ in range(model.n_students)]

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True)

            # Forward through model so projectors (if any) are applied
            teacher_features, students_features = model(images)

            for student_idx, (student, optimizer) in enumerate(
                zip(model.students, optimizers)
            ):
                student_features = students_features[student_idx]
                loss = student_teacher_loss(teacher_features, student_features)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_losses[student_idx].append(float(loss.item()))

        mean_train_losses = [
            float(np.mean(losses)) if len(losses) > 0 else 0.0
            for losses in train_losses
        ]
        train_loss = float(np.mean(mean_train_losses))

        calibrate_student_teacher_scores(
            model=model,
            loader=calib_loader,
            image_size=image_size,
            device=device,
        )

        val_metrics = evaluate_student_teacher(
            model=model,
            loader=val_loader,
            image_size=image_size,
            device=device,
            topk_percent=topk_percent,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        students_loss_str = " | ".join(
            [f"S{idx}={loss:.6f}" for idx, loss in enumerate(mean_train_losses)]
        )
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6f} ({students_loss_str}) | pixel_AP={val_metrics['pixel_ap']:.4f} | pixel_AUROC={val_metrics['pixel_auroc']:.4f} | best_thr={val_metrics['best_threshold']:.6f}"
        )

        if val_metrics["pixel_ap"] > best_pixel_ap:
            best_pixel_ap = val_metrics["pixel_ap"]
            best_state = {
                "students": [student.state_dict() for student in model.students],
                "student_projectors": [
                    p.state_dict() for p in getattr(model, "student_projectors", [])
                ],
                "teacher_backbone": backbone,
                "student_backbone": student_backbone,
                "out_indices": out_indices,
                "n_students": n_students,
                "image_size": image_size,
                "best_pixel_ap": best_pixel_ap,
                "best_threshold": val_metrics["best_threshold"],
                "score_stats": model.score_stats,
                "epoch": epoch,
            }

    if best_state is not None:
        for student, student_state in zip(model.students, best_state["students"]):
            student.load_state_dict(student_state)
        for proj, proj_state in zip(
            getattr(model, "student_projectors", []),
            best_state.get("student_projectors", []),
        ):
            proj.load_state_dict(proj_state)
        model.set_score_stats(best_state.get("score_stats"))

    sb_name = student_backbone if student_backbone is not None else backbone
    save_path = (
        ST_SAVE_DIR / f"student_teacher_{class_name}_T-{backbone}_S-{sb_name}.pt"
    )

    torch.save({"model": best_state, "history": history}, save_path)

    print(f"Saved best model to: {save_path}")
    print(f"Best validation Pixel AP: {best_pixel_ap:.4f}")

    return model, history, save_path


@torch.no_grad()
def evaluate_student_teacher(
    model: StudentTeacherAD,
    loader: DataLoader,
    image_size: int,
    device: str = DEVICE,
    topk_percent: float = 1.0,
):
    model.eval()

    all_scores = []
    all_masks = []
    image_scores = []
    image_labels = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        anomaly_maps = compute_student_teacher_maps(
            model=model, images=images, image_size=image_size
        )

        scores = anomaly_maps.detach().cpu().numpy()

        img_scores = (
            image_score_from_map(anomaly_maps, topk_percent=topk_percent)
            .detach()
            .cpu()
            .numpy()
        )

        all_scores.append(scores.reshape(-1))
        all_masks.append(masks.reshape(-1))

        image_scores.append(img_scores)
        image_labels.append(labels)

    y_score = np.concatenate(all_scores) if len(all_scores) > 0 else np.array([])
    y_true = (
        np.concatenate(all_masks).astype(np.uint8)
        if len(all_masks) > 0
        else np.array([])
    )

    img_score = np.concatenate(image_scores) if len(image_scores) > 0 else np.array([])
    img_true = (
        np.concatenate(image_labels).astype(np.uint8)
        if len(image_labels) > 0
        else np.array([])
    )

    pixel_ap = (
        average_precision_score(y_true, y_score)
        if y_true.size and y_score.size
        else 0.0
    )

    pixel_has_both_classes = y_true.size > 0 and np.unique(y_true).size > 1
    try:
        pixel_auroc = (
            roc_auc_score(y_true, y_score)
            if y_true.size and y_score.size and pixel_has_both_classes
            else float("nan")
        )
    except ValueError:
        pixel_auroc = float("nan")

    image_has_both_classes = img_true.size > 0 and np.unique(img_true).size > 1
    try:
        image_ap = (
            average_precision_score(img_true, img_score)
            if img_true.size and img_score.size
            else float("nan")
        )
        image_auroc = (
            roc_auc_score(img_true, img_score)
            if img_true.size and img_score.size and image_has_both_classes
            else float("nan")
        )
    except ValueError:
        image_ap = float("nan")
        image_auroc = float("nan")

    precision, recall, thresholds = (
        precision_recall_curve(y_true, y_score)
        if y_true.size and y_score.size
        else (np.array([]), np.array([]), np.array([]))
    )

    f1 = (
        2.0 * precision * recall / (precision + recall + 1e-8)
        if precision.size
        else np.array([])
    )

    if len(thresholds) > 0 and len(f1) > 0:
        best_idx = int(np.nanargmax(f1[:-1]))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])
    else:
        best_threshold = float(np.percentile(y_score, 99.5)) if y_score.size else 0.0
        best_f1 = 0.0

    return {
        "pixel_ap": float(pixel_ap),
        "pixel_auroc": float(pixel_auroc),
        "image_ap": float(image_ap),
        "image_auroc": float(image_auroc),
        "best_threshold": best_threshold,
        "best_pixel_f1": best_f1,
    }


# ===== Visualization helpers =====

import matplotlib.pyplot as plt


def denormalize_image_tensor(x: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    x = x.cpu() * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def normalize_map_np(m):
    m = np.asarray(m, dtype=np.float32)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


@torch.no_grad()
def show_student_teacher_predictions(
    model: StudentTeacherAD,
    dataset: PixelValDataset,
    n: int = 6,
    image_size: int = 224,
    device: str = DEVICE,
):
    model.eval()

    indices = np.linspace(0, len(dataset) - 1, min(n, len(dataset))).astype(int)

    for idx in indices:
        item = dataset[idx]

        image = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].numpy()

        anomaly_map = (
            compute_student_teacher_maps(
                model=model, images=image, image_size=image_size
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        img_np = denormalize_image_tensor(item["image"])
        map_np = normalize_map_np(anomaly_map)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(map_np, cmap="jet")
        axes[2].set_title("ST anomaly map")
        axes[2].axis("off")

        axes[3].imshow(img_np)
        axes[3].imshow(map_np, cmap="jet", alpha=0.45)
        axes[3].set_title(f"Overlay\n{item['anomaly_type']}")
        axes[3].axis("off")

        plt.suptitle(item["path"], fontsize=9)
        plt.tight_layout()
        plt.show()


# ===== Ensemble helper =====


def standardize_map(m):
    m = np.asarray(m, dtype=np.float32)
    med = np.median(m)
    q75 = np.percentile(m, 75)
    q25 = np.percentile(m, 25)
    iqr = q75 - q25
    return (m - med) / (iqr + 1e-8)


def combine_patchcore_student_teacher(
    patchcore_map, st_map, w_patchcore: float = 0.6, w_st: float = 0.4
):
    pc = standardize_map(patchcore_map)
    st = standardize_map(st_map)

    combined = w_patchcore * pc + w_st * st
    return combined
