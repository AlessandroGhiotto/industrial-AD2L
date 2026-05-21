from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

def tensor_to_numpy_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)

DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"

try:
    from transformers import AutoModelForImageSegmentation
    import torchvision.transforms as transforms
except ImportError as exc:  # pragma: no cover - dependency guard
    AutoModelForImageSegmentation = None
    transforms = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_pil_rgb(image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        array = tensor_to_numpy_image(image)
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _resolve_fill(
    fill: str | tuple[int, int, int] | int | float, image: np.ndarray
) -> np.ndarray:
    if isinstance(fill, str):
        fill_key = fill.lower()
        if fill_key == "white":
            value = np.array([255, 255, 255], dtype=np.float32)
        elif fill_key == "black":
            value = np.array([0, 0, 0], dtype=np.float32)
        elif fill_key == "gray":
            value = np.array([127, 127, 127], dtype=np.float32)
        elif fill_key == "mean":
            value = image.reshape(-1, image.shape[-1]).mean(axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported fill value: {fill}")
    elif isinstance(fill, (tuple, list, np.ndarray)):
        value = np.asarray(fill, dtype=np.float32)
        if value.size != 3:
            raise ValueError("RGB fill must have exactly 3 values")
    else:
        value = np.array([fill, fill, fill], dtype=np.float32)

    return value.reshape(1, 1, 3)


def _extract_mask_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        mask = outputs
    elif isinstance(outputs, dict):
        for key in ("logits", "preds", "prediction", "predictions", "mask"):
            if key in outputs and outputs[key] is not None:
                mask = outputs[key]
                break
        else:
            raise RuntimeError("BiRefNet output did not include logits/preds/mask")
    else:
        for key in ("logits", "preds", "prediction", "predictions", "mask"):
            if hasattr(outputs, key):
                value = getattr(outputs, key)
                if value is not None:
                    mask = value
                    break
        else:
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                mask = outputs[0]
            else:
                raise RuntimeError(
                    "Unable to extract a mask tensor from BiRefNet outputs"
                )

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.mean(dim=0)

    mask = mask.float()
    if mask.min().item() < 0.0 or mask.max().item() > 1.0:
        mask = torch.sigmoid(mask)
    return mask


@dataclass
class BiRefNetBackgroundRemover:
    model_name: str = DEFAULT_BIREFNET_MODEL
    input_size: int = 1024
    device: str | torch.device | None = None
    trust_remote_code: bool = True
    use_half_precision: bool = True

    def __post_init__(self) -> None:
        if AutoModelForImageSegmentation is None or transforms is None:
            raise RuntimeError(
                "transformers and torchvision are required for BiRefNet background removal"
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.device = _resolve_device(self.device)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self.model.eval()

        if self.use_half_precision and self.device.type == "cuda":
            torch.set_float32_matmul_precision(["high", "highest"][0])
            self.model.half()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def predict_mask(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        pil_image = _to_pil_rgb(image)
        original_size = (pil_image.height, pil_image.width)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        if self.use_half_precision and self.device.type == "cuda":
            input_tensor = input_tensor.half()

        outputs = self.model(input_tensor)
        # BiRefNet returns a tuple; the last element is the mask logits
        mask = outputs[-1].sigmoid().squeeze()
        mask = mask.float()

        if target_size is None:
            target_size = original_size

        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(
            mask,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return mask.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def remove_background(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        fill: str | tuple[int, int, int] | int | float = "white",
        threshold: float = 0.5,
        target_size: tuple[int, int] | None = None,
    ) -> Image.Image | torch.Tensor:
        pil_image = _to_pil_rgb(image)
        mask = self.predict_mask(pil_image, target_size=target_size)

        image_array = np.asarray(pil_image, dtype=np.float32)
        fill_array = _resolve_fill(fill, image_array)
        foreground = (mask >= float(threshold)).astype(np.float32)[..., None]
        blended = image_array * foreground + fill_array * (1.0 - foreground)
        blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        if torch.is_tensor(image):
            return torch.from_numpy(blended.astype(np.float32) / 255.0).permute(2, 0, 1)
        return Image.fromarray(blended)

    def remove_background_file(
        self,
        image_path: str | Path,
        output_path: str | Path,
        fill: str | tuple[int, int, int] | int | float = "white",
        threshold: float = 0.5,
    ) -> Path:
        image_path = Path(image_path)
        output_path = Path(output_path)
        image = Image.open(image_path).convert("RGB")
        cleaned = self.remove_background(
            image,
            fill=fill,
            threshold=threshold,
            target_size=(image.height, image.width),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(cleaned, torch.Tensor):
            array = (cleaned.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
            Image.fromarray(array.astype(np.uint8)).save(output_path)
        else:
            cleaned.save(output_path)
        return output_path


class ForegroundRemovedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        remover: BiRefNetBackgroundRemover,
        fill: str | tuple[int, int, int] | int | float = "white",
        threshold: float = 0.5,
    ) -> None:
        self.base_dataset = base_dataset
        self.remover = remover
        self.fill = fill
        self.threshold = float(threshold)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = dict(self.base_dataset[idx])
        if "image" not in sample:
            raise KeyError(
                "ForegroundRemovedDataset expects each sample to contain 'image'"
            )
        sample["image"] = self.remover.remove_background(
            sample["image"],
            fill=self.fill,
            threshold=self.threshold,
        )
        return sample


def preprocess_dataset_tree(
    input_root: str | Path,
    output_root: str | Path,
    remover: BiRefNetBackgroundRemover,
    fill: str | tuple[int, int, int] | int | float = "white",
    threshold: float = 0.5,
    image_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> list[Path]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    image_suffixes = {suffix.lower() for suffix in image_suffixes}

    saved_paths: list[Path] = []
    for image_path in sorted(input_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
            continue
        if "ground_truth_train" in image_path.parts:
            continue

        relative_path = image_path.relative_to(input_root)
        output_path = output_root / relative_path
        remover.remove_background_file(
            image_path=image_path,
            output_path=output_path,
            fill=fill,
            threshold=threshold,
        )
        saved_paths.append(output_path)

    return saved_paths
