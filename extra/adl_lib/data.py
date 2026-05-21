from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from adl_lib.config import SEED


def list_image_files(folder):
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_numpy_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)


def tensor_to_numpy_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    return x.astype(np.float32)


class ADLTrainGoodDataset(Dataset):
    """Normal training images only: class_XX/train/good"""

    def __init__(self, root, class_name, image_size=256):
        self.root = Path(root)
        self.class_name = class_name
        self.image_size = int(image_size)

        good_dir = self.root / class_name / "train" / "good"
        self.samples = list_image_files(good_dir)

        if len(self.samples) == 0:
            raise RuntimeError(f"No training good images found in: {good_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        return {
            "image": pil_to_tensor(image),
            "path": str(img_path),
        }


class ADLTrainAnomalyLabeledDataset(Dataset):
    """Labeled anomaly examples from class_XX/train/anomaly_YY + class_XX/ground_truth_train/anomaly_YY."""

    def __init__(self, root, class_name, image_size=256):
        self.root = Path(root)
        self.class_name = class_name
        self.image_size = int(image_size)

        train_root = self.root / class_name / "train"
        gt_root = self.root / class_name / "ground_truth_train"

        self.samples = []
        anomaly_dirs = sorted(
            [
                d
                for d in train_root.iterdir()
                if d.is_dir() and d.name.startswith("anomaly_")
            ]
        )

        for a_dir in anomaly_dirs:
            gt_dir = gt_root / a_dir.name
            img_files = list_image_files(a_dir)
            for img_path in img_files:
                mask_path = gt_dir / img_path.name
                if not mask_path.exists():
                    raise RuntimeError(f"Missing mask for {img_path.name} in {gt_dir}")
                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "label": 1,
                        "defect_type": a_dir.name,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"No anomaly samples found under {train_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        image = Image.open(rec["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask = Image.open(rec["mask_path"]).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = (pil_to_tensor(mask) > 0.5).float()

        return {
            "image": pil_to_tensor(image),
            "mask": mask,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "defect_type": rec["defect_type"],
            "path": str(rec["image_path"]),
        }


class ADLTestUnlabeledDataset(Dataset):
    """Unlabeled leaderboard images from class_XX/test."""

    def __init__(self, root, class_name, image_size=256):
        self.root = Path(root)
        self.class_name = class_name
        self.image_size = int(image_size)

        test_dir = self.root / class_name / "test"
        self.samples = list_image_files(test_dir)

        if len(self.samples) == 0:
            raise RuntimeError(f"No test images found in: {test_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        return {
            "image": pil_to_tensor(image),
            "path": str(img_path),
        }


def make_labeled_eval_from_train(
    root, class_name, image_size=256, val_good_ratio=0.15, seed=SEED
):
    good_ds = ADLTrainGoodDataset(root, class_name, image_size=image_size)
    anom_ds = ADLTrainAnomalyLabeledDataset(root, class_name, image_size=image_size)

    n_good = len(good_ds)
    idx = np.arange(n_good)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val_good = max(1, int(round(n_good * val_good_ratio)))
    val_good_idx = idx[:n_val_good].tolist()
    train_good_idx = idx[n_val_good:].tolist()

    train_good_subset = Subset(good_ds, train_good_idx)
    val_good_subset = Subset(good_ds, val_good_idx)

    return train_good_subset, val_good_subset, anom_ds


class DictDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
