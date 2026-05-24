"""Train + evaluate the convolutional autoencoder baseline on all 8 classes.

Ported from `final/extra/autoenc1-2.ipynb` to a local-runnable script (the
notebook is Colab-only — paths and torchmetrics install removed). The model is
a 15-channel multi-view conv-AE trained with a hybrid MSE+SSIM loss, exactly
as in the notebook's final cell.

At eval time the per-view anomaly map (mean absolute reconstruction error,
gaussian-smoothed) is scored against the GT mask for each view image
independently — same protocol as the other rows of Table 1.

Checkpoints land in `outputs/autoencoder_models/`. Per-class APs in
`outputs/table_results/autoencoder.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
sys.path.insert(0, str(FINAL_DIR))

from src import config_final as cfg  # noqa: E402
from src.eval_table import evaluate_all_classes, save_results  # noqa: E402


# ── Hyperparameters (copied from the notebook) ────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
NUM_VIEWS = 5
TRAIN_SPLIT = 0.9
ANOMALY_MAP_SIGMA = 4.0
SEED = 7

AE_DATASET = cfg.DATASET_DIR  # BiRefNet-preprocessed (same as other rows except U-Net⁻)
AE_SAVE_DIR = cfg.OUTPUT_DIR / "autoencoder_models"
OUT_RESULTS = cfg.OUTPUT_DIR / "table_results" / "autoencoder.json"

VIEW_SUFFIXES = [f"_view{i}" for i in range(1, NUM_VIEWS + 1)]


# ── Utilities ─────────────────────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_rgb_image(path: Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return pil_to_tensor(img)


def remove_view_suffix(filename: str) -> str:
    stem = Path(filename).stem
    for suffix in VIEW_SUFFIXES:
        if stem.endswith(suffix):
            return stem.replace(suffix, "")
    return stem


def group_multiview_samples(folder: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for img_path in sorted(Path(folder).glob("*.png")):
        sample_id = remove_view_suffix(img_path.name)
        view_key = img_path.stem.split("_")[-1]  # 'view1'..'view5'
        grouped.setdefault(sample_id, {})[view_key] = img_path
    return grouped


class ADLGoodMultiView(Dataset):
    """5 good-view images per sample, stacked into a 15-channel tensor."""

    def __init__(self, root: Path, class_name: str, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: list[list[Path]] = []
        good_dir = Path(root) / class_name / "train" / "good"
        grouped = group_multiview_samples(good_dir)
        for view_dict in grouped.values():
            views = [view_dict.get(f"view{i}") for i in range(1, NUM_VIEWS + 1)]
            if all(v is not None for v in views):
                self.samples.append(views)
        if len(self.samples) == 0:
            raise RuntimeError(f"No multi-view samples in {good_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        views = [load_rgb_image(p, self.image_size) for p in self.samples[idx]]
        return torch.cat(views, dim=0)  # (15, H, W)


# ── Model (verbatim from the notebook) ────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels: int = 15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── SSIM (lazy import; torchmetrics required) ─────────────────────────────────
def get_ssim_module(device: str):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def hybrid_loss(recon, target, ssim_module, alpha: float = 0.5):
    mse = F.mse_loss(recon, target)
    ssim_val = ssim_module(recon, target)
    return alpha * mse + (1 - alpha) * (1 - ssim_val)


# ── Train / load ──────────────────────────────────────────────────────────────
def checkpoint_path(class_name: str) -> Path:
    return AE_SAVE_DIR / f"ae_{class_name}.pt"


def train_one_class(class_name: str, device: str) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = ADLGoodMultiView(AE_DATASET, class_name, image_size=IMAGE_SIZE)
    n_train = int(TRAIN_SPLIT * len(ds))
    n_val = len(ds) - n_train
    if n_val < 1:
        n_val = 1
        n_train = len(ds) - 1
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvAutoencoder(input_channels=NUM_VIEWS * 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ssim_module = get_ssim_module(device)

    best_val = float("inf")
    ckpt = checkpoint_path(class_name)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = hybrid_loss(model(x), x, ssim_module)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                val_loss += hybrid_loss(model(x), x, ssim_module).item()
        val_loss /= max(1, len(val_loader))

        print(f"[{class_name}] epoch {epoch:02d}/{EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, ckpt)


def train_all(classes: list[str], device: str) -> None:
    AE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        if checkpoint_path(cls).exists():
            print(f"[{cls}] checkpoint exists, skipping training")
            continue
        print(f"\n=== Training Autoencoder on {cls} ===")
        train_one_class(cls, device)


# ── Inference helper for the eval table ───────────────────────────────────────
def _find_view_siblings(img_path: Path) -> list[Path] | None:
    """Return the 5 view images for the sample containing `img_path`, or None."""
    stem = remove_view_suffix(img_path.name)
    siblings = [img_path.parent / f"{stem}_view{i}.png" for i in range(1, NUM_VIEWS + 1)]
    return siblings if all(p.exists() for p in siblings) else None


def _view_index(img_path: Path) -> int:
    stem = img_path.stem
    for i, suf in enumerate(VIEW_SUFFIXES):
        if stem.endswith(suf):
            return i
    return 0


def make_predict_fn_factory(device: str):
    def factory(class_name: str):
        ckpt = checkpoint_path(class_name)
        model = ConvAutoencoder(input_channels=NUM_VIEWS * 3).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()

        @torch.no_grad()
        def predict_fn(img_path: Path, img_pil: Image.Image) -> np.ndarray:
            siblings = _find_view_siblings(img_path)
            view_idx = _view_index(img_path)
            if siblings is None:
                # Fall back: replicate the same image into all 5 views.
                t = load_rgb_image(img_path, IMAGE_SIZE)
                x = torch.cat([t] * NUM_VIEWS, dim=0).unsqueeze(0).to(device)
            else:
                views = [load_rgb_image(p, IMAGE_SIZE) for p in siblings]
                x = torch.cat(views, dim=0).unsqueeze(0).to(device)
            recon = model(x)
            # Per-view anomaly map: mean |error| over the 3 channels of THIS view
            c0 = view_idx * 3
            err = (x[:, c0:c0 + 3] - recon[:, c0:c0 + 3]).abs().mean(dim=1)[0]
            amap = err.detach().cpu().numpy().astype(np.float32)
            return gaussian_filter(amap, sigma=ANOMALY_MAP_SIGMA)
        return predict_fn
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(
        d.name for d in AE_DATASET.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    )
    print(f"Classes: {classes}")

    if not args.skip_train:
        train_all(classes, device)

    print(f"\n=== Evaluating Autoencoder ===")
    results = evaluate_all_classes(
        predict_fn_factory=make_predict_fn_factory(device),
        dataset_dir=AE_DATASET,
        classes=classes,
    )
    save_results(results, OUT_RESULTS, method_name="Autoencoder")


if __name__ == "__main__":
    main()
