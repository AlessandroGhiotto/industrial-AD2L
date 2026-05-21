"""WinCLIP+ inference pipeline.

Per class:
  1. Build text prompts (normal templates × object_name, plus per-anomaly
     descriptions from anomaly_descriptions.csv).
  2. Fit a CLIP-Large patch-feature memory bank on the train-good split.
  3. Fit per-defect CLS features on labeled anomalies (for semantic search at
     inference: each test patch grid is reweighted toward the text embedding
     of the NEAREST labeled anomaly).
  4. Predict anomaly maps on the labeled, val-good, and test pools.
  5. Dump raw maps for ensembling AND write a standalone submission.

No gradient updates anywhere — this is a frozen-CLIP inference pipeline.

Output layout matches the other models so that ensemble_weighted.py can pick
this up with `--models winclip`.
"""
import sys
sys.path.append('.')
import argparse
import csv
import zipfile
from pathlib import Path as P

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from adl_lib.config import (
    PATH, IMAGE_SIZE, BATCH_SIZE, SEED, seed_everything,
)
from adl_lib.data import (
    ADLTestUnlabeledDataset, make_labeled_eval_from_train,
)
from adl_lib.utils import float_matrix_to_q8rle
from adl_lib.dump import dump_labeled, dump_test, dump_good_eval
from adl_lib.winclip import WinCLIPPlus


# CLIP normalization (OpenAI ViT mean/std). Inputs are in [0, 1].
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


CLASS_TO_OBJECT = {
    "class_01": "resistor",
    "class_02": "inductor",
    "class_03": "gear",
    "class_04": "screw",
    "class_05": "nut",
    "class_06": "coffee",
    "class_07": "pistachio",
    "class_08": "capsule",
}


def _to_clip_inputs(img_batch, device):
    """img_batch: (B, 3, H, W) in [0, 1]. Returns normalized tensor on device."""
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)
    return (img_batch.to(device) - mean) / std


def build_defect_to_desc_idx(csv_path, class_name, anomaly_descriptions):
    """Map 'anomaly_NN' → index into model.anomaly_descriptions list.

    The model's `anomaly_descriptions` is the unique-per-class description
    list (with a generic catch-all appended at the end). We build the lookup
    by reading the CSV.
    """
    df = pd.read_csv(csv_path)
    class_df = df[df['public_class'] == class_name]
    desc_to_idx = {d: i for i, d in enumerate(anomaly_descriptions)}
    generic_idx = len(anomaly_descriptions) - 1
    mapping = {}
    for _, row in class_df.iterrows():
        anom = row['public_anomaly']
        desc = row['description']
        if isinstance(desc, str) and desc in desc_to_idx:
            mapping[anom] = desc_to_idx[desc]
        else:
            mapping[anom] = generic_idx
    return mapping


@torch.no_grad()
def predict_pool(model, dataset, batch_size, device, alpha=0.7, image_size=IMAGE_SIZE,
                 include_mask=False, desc=""):
    """Run WinCLIP+ on a dataset.

    Returns a list of dicts with keys: anomaly_map (H, W), path, (mask, label).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True)
    out = []
    for batch in tqdm(loader, desc=desc):
        images = _to_clip_inputs(batch["image"], device)
        amap = model(images, alpha=alpha)  # (B, 1, H, W)
        if amap.shape[-1] != image_size:
            amap = F.interpolate(amap, size=(image_size, image_size),
                                 mode="bilinear", align_corners=False)
        amap = amap.squeeze(1).float().cpu().numpy()  # (B, H, W)
        for i in range(len(batch["path"])):
            item = {"anomaly_map": amap[i], "path": batch["path"][i]}
            if include_mask:
                item["mask"] = batch["mask"][i].squeeze().cpu().numpy().astype(np.uint8)
                item["label"] = int(batch["label"][i])
            out.append(item)
    return out


def encode_submission_rows(test_results, p_low=1.0, p_high=99.5):
    """Per-class percentile clipping → [0, 1] → q8rle."""
    rows = []
    flat = np.concatenate([r['anomaly_map'].ravel() for r in test_results])
    if flat.size == 0:
        return rows
    lo = float(np.percentile(flat, p_low))
    hi = float(np.percentile(flat, p_high))
    for r in test_results:
        amap = r['anomaly_map']
        if hi > lo:
            sub = np.clip((amap - lo) / (hi - lo), 0.0, 1.0)
        else:
            sub = np.zeros_like(amap)
        rows.append({
            'ID': P(r['path']).stem,
            'Label': float_matrix_to_q8rle(sub),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="WinCLIP+ inference + dump + submission.")
    parser.add_argument("--classes", type=str, nargs="+",
                        help="Subset of classes (default: all under PATH).")
    parser.add_argument("--backbone", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--csv_path", type=str,
                        default="dataset/adl-2025-2026-anomaly-detection/anomaly_descriptions.csv")
    parser.add_argument("--dump_raw", type=str, default="artifacts/raw_outputs")
    parser.add_argument("--model_name", type=str, default="winclip")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Memory weight in fusion: amap = (1-alpha)*text + alpha*memory.")
    parser.add_argument("--val_good_ratio", type=float, default=0.15)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--memory_max_samples", type=int, default=25000)
    parser.add_argument("--submission_tag", type=str, default="winclip")
    args = parser.parse_args()

    seed_everything(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Backbone: {args.backbone}")

    if args.classes:
        classes = args.classes
    else:
        classes = sorted([d.name for d in P(PATH).iterdir()
                          if d.is_dir() and d.name.startswith("class_")])
    print(f"Classes: {classes}")

    submission_rows = []

    for class_name in classes:
        print(f"\n===== {class_name} =====")
        object_name = CLASS_TO_OBJECT.get(class_name, "object")
        print(f"object_name: {object_name}")

        train_good, val_good, anom_ds = make_labeled_eval_from_train(
            PATH, class_name, image_size=args.image_size,
            val_good_ratio=args.val_good_ratio, seed=SEED,
        )
        test_ds = ADLTestUnlabeledDataset(PATH, class_name, image_size=args.image_size)
        print(f"  train_good={len(train_good)}  val_good={len(val_good)}  "
              f"anom={len(anom_ds)}  test={len(test_ds)}")

        # Build a fresh model per class (text prompts + memory bank are class-specific).
        model = WinCLIPPlus(model_name=args.backbone, device=device)
        model.setup_prompts(class_name, object_name, args.csv_path)
        print(f"  prompts: {len(model.anomaly_descriptions)} anomaly descriptions")

        # Normal memory bank.
        train_loader = DataLoader(train_good, batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)
        # The fit_normal in winclip.py expects pre-normalized inputs in batch["image"].
        # Wrap the loader so we hand over normalized tensors.
        class _NormWrap:
            def __init__(self, loader, device):
                self.loader = loader
                self.device = device
            def __iter__(self):
                for batch in self.loader:
                    yield {"image": _to_clip_inputs(batch["image"], "cpu")}
            def __len__(self):
                return len(self.loader)
        model.fit_normal(_NormWrap(train_loader, device),
                         max_samples=args.memory_max_samples)

        # Labeled anomaly CLS features.
        anom_loader = DataLoader(anom_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
        class _NormWrapAnom:
            def __init__(self, loader, device):
                self.loader = loader
                self.device = device
            def __iter__(self):
                for batch in self.loader:
                    yield {"image": _to_clip_inputs(batch["image"], "cpu"),
                           "defect_type": batch["defect_type"]}
            def __len__(self):
                return len(self.loader)
        model.fit_labeled_anomalies(_NormWrapAnom(anom_loader, device))

        # Populate defect_to_desc_idx so semantic search can find the matching text.
        model.defect_to_desc_idx = build_defect_to_desc_idx(
            args.csv_path, class_name, model.anomaly_descriptions,
        )

        # Predict on the three pools.
        labeled_results = predict_pool(
            model, anom_ds, args.batch_size, device,
            alpha=args.alpha, image_size=args.image_size,
            include_mask=True, desc=f"{class_name}/labeled",
        )
        good_results = predict_pool(
            model, val_good, args.batch_size, device,
            alpha=args.alpha, image_size=args.image_size,
            include_mask=False, desc=f"{class_name}/good",
        )
        test_results = predict_pool(
            model, test_ds, args.batch_size, device,
            alpha=args.alpha, image_size=args.image_size,
            include_mask=False, desc=f"{class_name}/test",
        )

        if args.dump_raw:
            dump_labeled(args.dump_raw, args.model_name, class_name, labeled_results)
            dump_good_eval(args.dump_raw, args.model_name, class_name, good_results)
            dump_test(args.dump_raw, args.model_name, class_name, test_results)
            print(f"  dumped {args.dump_raw}/{args.model_name}/{class_name}/")

        submission_rows.extend(encode_submission_rows(test_results))

        del model
        torch.cuda.empty_cache()

    # Standalone submission.
    sub_dir = P("submission")
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / f"submission_{args.submission_tag}.csv"
    if submission_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Label"])
            writer.writeheader()
            writer.writerows(submission_rows)
    zip_path = sub_dir / f"submission_{args.submission_tag}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
    print(f"\nWrote {csv_path} and {zip_path}")


if __name__ == "__main__":
    main()
