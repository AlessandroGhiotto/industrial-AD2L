"""Train a U-Net per class on the RAW (non-BiRefNet) dataset and write per-class pixel-AP.

This is the ablation row "U-Net w/o BiRefNet" in the report table. The only
change vs the main U-Net pipeline is `dataset_dir`: we point it at
`dataset/adl-2025-2026-anomaly-detection` (no background removal) instead of
the `_birefnet` variant. Checkpoints land under `outputs/unet_models_nobirefnet/`
so they don't clobber the main run.

Usage:
    python final/scripts/train_eval_unet_no_birefnet.py
    python final/scripts/train_eval_unet_no_birefnet.py --skip-train  # eval only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make `final/` importable
REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
sys.path.insert(0, str(FINAL_DIR))

from src import config_final as cfg
from src import unet_final as unet
from src.eval_table import evaluate_all_classes, save_results


RAW_DATASET = REPO_ROOT / "dataset" / "adl-2025-2026-anomaly-detection"
OUT_MODEL_DIR = cfg.OUTPUT_DIR / "unet_models_nobirefnet"
OUT_RESULTS = cfg.OUTPUT_DIR / "table_results" / "unet_no_birefnet.json"


def list_classes() -> list[str]:
    return sorted(
        d.name for d in RAW_DATASET.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    )


def train_all(classes: list[str]) -> None:
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        ckpt = OUT_MODEL_DIR / cls / "best.pt"
        if ckpt.exists():
            print(f"[{cls}] checkpoint exists, skipping training: {ckpt}")
            continue
        print(f"\n=== Training U-Net (no BiRefNet) on {cls} ===")
        unet.train_one_class(
            dataset_dir=str(RAW_DATASET),
            class_name=cls,
            desc_csv=str(RAW_DATASET / "anomaly_descriptions.csv"),
            out_dir=str(OUT_MODEL_DIR),
            image_size=cfg.UNET_IMAGE_SIZE,
            batch_size=cfg.UNET_BATCH_SIZE,
            epochs=cfg.UNET_EPOCHS,
            lr=cfg.UNET_LR,
            val_ratio=cfg.UNET_VAL_RATIO,
            p_synth=cfg.UNET_P_SYNTH,
            p_cutpaste=cfg.UNET_P_CUTPASTE,
            p_multi_paste=cfg.UNET_P_MULTI_PASTE,
            seed=cfg.SEED,
            encoder=cfg.UNET_ENCODER,
            use_attention_gates=cfg.UNET_USE_ATTENTION_GATES,
            pose_json=str(cfg.UNET_POSE_JSON) if cfg.UNET_POSE_JSON.exists() else None,
            use_shape_bank=cfg.UNET_USE_SHAPE_BANK,
        )


def make_predict_fn_factory(device: torch.device):
    """Return a factory that loads one U-Net per class and yields a predict_fn."""
    def factory(class_name: str):
        ckpt = OUT_MODEL_DIR / class_name / "best.pt"
        model = unet.load_model(str(ckpt), device)
        img_size = cfg.UNET_IMAGE_SIZE

        def predict_fn(img_path: Path, img_pil: Image.Image) -> np.ndarray:
            view_id = unet._parse_view_and_id(img_path.name)[1]
            v_t = torch.tensor([view_id], device=device, dtype=torch.long)
            img_rs = img_pil.resize((img_size, img_size))
            prob = unet._infer_single_tta(model, img_rs, v_t, device,
                                          use_tta=cfg.UNET_USE_TTA)
            return prob  # already (img_size, img_size)
        return predict_fn
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; just evaluate existing checkpoints.")
    args = parser.parse_args()

    assert RAW_DATASET.exists(), f"Raw dataset not found: {RAW_DATASET}"
    classes = list_classes()
    print(f"Classes: {classes}")

    if not args.skip_train:
        train_all(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Evaluating U-Net (no BiRefNet) ===")
    results = evaluate_all_classes(
        predict_fn_factory=make_predict_fn_factory(device),
        dataset_dir=RAW_DATASET,
        classes=classes,
    )
    save_results(results, OUT_RESULTS, method_name="U-Net w/o BiRefNet")


if __name__ == "__main__":
    main()
