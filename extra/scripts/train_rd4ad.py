"""
Train RD4AD (Reverse Distillation) on all classes.

Usage:
    python scripts/train_rd4ad.py
    python scripts/train_rd4ad.py --classes class_01 class_02
    python scripts/train_rd4ad.py --epochs 50 --force
"""

import sys
sys.path.append(".")

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from adl_lib.config import (
    PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    RD4AD_EPOCHS,
    seed_everything,
)
from adl_lib.data import (
    ADLTrainGoodDataset,
    ADLTrainAnomalyLabeledDataset,
    make_labeled_eval_from_train,
)
from adl_lib.rd4ad import RD4AD
from adl_lib.utils import normalize_map


def evaluate_on_labeled(model, loader):
    """Compute pixel-AP and pixel-AUROC on labeled anomaly data."""
    results = model.predict_labeled(loader)
    if len(results) == 0:
        return {"pixel_ap": float("nan"), "pixel_auroc": float("nan")}

    pixel_masks = np.concatenate(
        [r["mask"].reshape(-1) for r in results]
    ).astype(np.uint8)
    pixel_scores = np.concatenate(
        [r["anomaly_map"].reshape(-1) for r in results]
    ).astype(np.float32)

    if np.unique(pixel_masks).size < 2:
        return {"pixel_ap": float("nan"), "pixel_auroc": float("nan")}

    return {
        "pixel_ap": float(average_precision_score(pixel_masks, pixel_scores)),
        "pixel_auroc": float(roc_auc_score(pixel_masks, pixel_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RD4AD on anomaly detection dataset")
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Specific classes to train (e.g. class_01 class_02). Default: all."
    )
    parser.add_argument("--epochs", type=int, default=RD4AD_EPOCHS, help="Training epochs")
    parser.add_argument("--force", action="store_true", help="Force retrain even if checkpoint exists")
    parser.add_argument("--dataset", type=str, default=PATH, help="Dataset root path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    seed_everything(SEED)

    dataset_path = args.dataset
    print(f"Dataset: {dataset_path}")

    # Discover classes
    if args.classes:
        classes = args.classes
    else:
        classes = sorted([
            d.name for d in Path(dataset_path).iterdir()
            if d.is_dir() and d.name.startswith("class_")
        ])

    print(f"Classes to train: {classes}")
    all_metrics = {}

    for class_name in classes:
        print(f"\n{'='*60}")
        print(f"  CLASS: {class_name}")
        print(f"{'='*60}")

        model = RD4AD(epochs=args.epochs)

        # Check for existing checkpoint
        if not args.force and model._load_checkpoint(class_name):
            print(f"Loaded existing checkpoint for {class_name}, skipping training.")
        else:
            # Prepare training data (all good images, no validation split)
            train_good_subset, _, _ = make_labeled_eval_from_train(
                dataset_path, class_name,
                image_size=IMAGE_SIZE,
                val_good_ratio=0.0,
                seed=SEED,
            )

            train_loader = DataLoader(
                train_good_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=False,
            )

            # Prepare validation loader (labeled anomalies)
            try:
                anom_ds = ADLTrainAnomalyLabeledDataset(
                    dataset_path, class_name, image_size=IMAGE_SIZE
                )
                val_loader = DataLoader(
                    anom_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )
            except RuntimeError:
                val_loader = None
                print(f"No labeled anomalies found for {class_name}, training without validation.")

            # Train
            model.fit(train_loader, class_name=class_name, val_loader=val_loader)

        # Evaluate on labeled anomalies
        try:
            anom_ds = ADLTrainAnomalyLabeledDataset(
                dataset_path, class_name, image_size=IMAGE_SIZE
            )
            eval_loader = DataLoader(
                anom_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            metrics = evaluate_on_labeled(model, eval_loader)
            all_metrics[class_name] = metrics
            print(f"\n  {class_name} → pixel_AP={metrics['pixel_ap']:.4f}, pixel_AUROC={metrics['pixel_auroc']:.4f}")
        except Exception as e:
            print(f"  Evaluation failed for {class_name}: {e}")
            all_metrics[class_name] = {"pixel_ap": float("nan"), "pixel_auroc": float("nan")}

        # Free GPU memory between classes
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY: RD4AD Results")
    print(f"{'='*60}")
    aps = []
    aurocs = []
    for cls_name, m in all_metrics.items():
        ap = m["pixel_ap"]
        auroc = m["pixel_auroc"]
        aps.append(ap)
        aurocs.append(auroc)
        print(f"  {cls_name}: pixel_AP={ap:.4f}, pixel_AUROC={auroc:.4f}")

    valid_aps = [a for a in aps if not np.isnan(a)]
    valid_aurocs = [a for a in aurocs if not np.isnan(a)]
    if valid_aps:
        print(f"\n  Mean pixel_AP:    {np.mean(valid_aps):.4f}")
        print(f"  Mean pixel_AUROC: {np.mean(valid_aurocs):.4f}")


if __name__ == "__main__":
    main()
