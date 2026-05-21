import sys
sys.path.append('.')
import argparse
import csv
import os
import zipfile
from itertools import cycle
from pathlib import Path as P

import numpy as np
import torch
import torch.nn.functional as F
# pandas intentionally not imported — use csv module to avoid pandas.io.formats.csvs bug
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from adl_lib.config import (
    PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED, seed_everything,
)
from adl_lib.data import (
    ADLTrainAnomalyLabeledDataset, ADLTrainGoodDataset, ADLTestUnlabeledDataset,
    make_labeled_eval_from_train,
)
from adl_lib.utils import (
    summarize_metrics, postprocess_anomaly_map, save_results_to_pdf, float_matrix_to_q8rle,
)
from adl_lib.simplenet import SimpleNetAD
from adl_lib.ensemble import apply_sample_gate
from adl_lib.dump import dump_labeled, dump_test, dump_good_eval


# ---------- losses ----------

def truncated_l1_loss(logits, labels, th=0.5):
    """Paper's truncated L1: push fake logits >= th, clean logits <= -th."""
    pos = labels > 0.5
    neg = ~pos
    loss_pos = torch.clamp(th - logits[pos], min=0).mean()
    loss_neg = torch.clamp(th + logits[neg], min=0).mean()
    return loss_pos + loss_neg


def margin_mask_loss(score_map, mask, margin=0.5):
    """Inside-mask scores should exceed outside-mask scores by at least `margin`."""
    pos_sum = (score_map * mask).sum(dim=(1, 2, 3))
    neg_sum = (score_map * (1 - mask)).sum(dim=(1, 2, 3))
    pos_cnt = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    neg_cnt = (1 - mask).sum(dim=(1, 2, 3)).clamp(min=1.0)
    return F.relu(margin - (pos_sum / pos_cnt - neg_sum / neg_cnt)).mean()


# ---------- training ----------

def train_simplenet(
    model, train_loader, anom_loader=None,
    epochs=15, lr=1e-3, device="cuda", th=0.5, w_aux=0.05,
):
    model.to(device)
    optimizer = Adam(
        list(model.adapter.parameters()) + list(model.discriminator.parameters()),
        lr=lr,
    )

    anom_iter = cycle(anom_loader) if anom_loader is not None else None

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            logits, labels = model(images)
            loss_main = truncated_l1_loss(logits, labels, th=th)
            loss = loss_main

            if anom_iter is not None and w_aux > 0:
                aux_batch = next(anom_iter)
                aux_images = aux_batch["image"].to(device)
                aux_masks = aux_batch["mask"].to(device)
                aux_logits = model.supervised_logits(aux_images)  # [B, 1, h, w]
                # Downsample mask to feature resolution for the margin loss.
                aux_masks_ds = F.interpolate(
                    aux_masks, size=aux_logits.shape[-2:], mode="nearest"
                )
                loss = loss + w_aux * margin_mask_loss(aux_logits, aux_masks_ds, margin=th)

            loss.backward()
            optimizer.step()
            pbar.set_postfix({"main": f"{loss_main.item():.4f}"})
    return model


def evaluate_simplenet(model, val_loader, device="cuda"):
    model.eval()
    model.to(device)
    results = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()
            anomaly_maps = model(images).squeeze(1).cpu().numpy()
            for i in range(len(anomaly_maps)):
                am = postprocess_anomaly_map(anomaly_maps[i], sigma=4.0)
                results.append({
                    "anomaly_map": am,
                    "score": float(np.quantile(am, 0.999)),
                    "mask": masks[i],
                    "label": int(labels[i]),
                    "image": images[i].cpu(),
                    "path": batch["path"][i],
                })
    return results


def predict_test(model, test_loader, device="cuda"):
    model.eval()
    model.to(device)
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting Test", leave=False):
            images = batch["image"].to(device)
            paths = batch["path"]
            anomaly_maps = model(images).squeeze(1).cpu().numpy()
            for i in range(len(anomaly_maps)):
                am = postprocess_anomaly_map(anomaly_maps[i], sigma=4.0)
                results.append({"anomaly_map": am, "path": paths[i]})
    return results


# ---------- submission helper ----------

def encode_submission_rows(test_results, p_low=1.0, p_high=99.5):
    """Per-class percentile clipping → [0, 1] → q8rle (preserves dynamic range)."""
    all_maps = [r['anomaly_map'] for r in test_results]
    if not all_maps:
        return []
    flat = np.concatenate([m.ravel() for m in all_maps])
    lo = float(np.percentile(flat, p_low))
    hi = float(np.percentile(flat, p_high))
    rows = []
    for r in test_results:
        amap = r['anomaly_map']
        if hi > lo:
            submission_map = np.clip((amap - lo) / (hi - lo), 0.0, 1.0)
        else:
            submission_map = np.zeros_like(amap)
        rows.append({
            'ID': P(r['path']).stem,
            'Label': float_matrix_to_q8rle(submission_map),
        })
    return rows


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SimpleNet on all classes.")
    parser.add_argument("--classes", type=str, nargs="+")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.015)
    parser.add_argument("--th", type=float, default=0.5, help="Truncated-L1 hinge threshold.")
    parser.add_argument("--w_aux", type=float, default=0.05,
                        help="Weight on auxiliary supervised mask loss; 0 disables.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--view_gate_alpha", type=float, default=0.5,
                        help="Multiplier strength for the sample-level multi-view gate; 0 disables.")
    parser.add_argument("--output_dir", type=str, default="artifacts/simplenet_v2")
    parser.add_argument("--dump_raw", type=str, default="artifacts/raw_outputs",
                        help="Root dir for raw anomaly-map dumps (for ensembling). "
                             "Empty string disables.")
    parser.add_argument("--model_name", type=str, default="simplenet")
    args = parser.parse_args()

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.classes:
        classes = args.classes
    else:
        classes = sorted([d.name for d in P(PATH).iterdir() if d.is_dir() and d.name.startswith("class_")])
    print(f"Processing {len(classes)} classes using SimpleNet ({args.backbone})")

    all_metrics, pdf_results, submission_rows = [], [], []

    for class_name in classes:
        print(f"\n>>> Class: {class_name}")
        model = SimpleNetAD(backbone_name=args.backbone, noise_std=args.noise_std).to(device)

        # Held-out good split. train_good_subset is used for training; val_good_subset
        # provides genuine held-out good predictions for the ensemble weight-fitter.
        train_good_subset, val_good_subset, _ = make_labeled_eval_from_train(
            PATH, class_name, image_size=IMAGE_SIZE,
            val_good_ratio=0.15, seed=SEED,
        )
        good_loader = DataLoader(train_good_subset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        anom_ds = ADLTrainAnomalyLabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        anom_loader_train = DataLoader(
            anom_ds, batch_size=min(args.batch_size, max(2, len(anom_ds))),
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        print(f"Training on {len(good_ds)} good + {len(anom_ds)} labeled anomaly samples...")
        train_simplenet(
            model, good_loader,
            anom_loader=anom_loader_train if args.w_aux > 0 else None,
            epochs=args.epochs, lr=args.lr, device=device,
            th=args.th, w_aux=args.w_aux,
        )

        val_loader = DataLoader(anom_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Evaluating on {len(anom_ds)} labeled anomaly samples (NOTE: training-set overlap)...")
        results = evaluate_simplenet(model, val_loader, device=device)

        metrics = summarize_metrics(results)
        metrics['class'] = class_name
        all_metrics.append(metrics)
        print(f"[train-set eval] Pixel AP: {metrics['pixel_ap']:.4f} | Pixel AUROC: {metrics['pixel_auroc']:.4f}")

        thr = metrics.get('pixel_threshold', 0.5)
        if np.isnan(thr):
            thr = 0.5
        pdf_results.append((class_name, results, thr))

        test_ds = ADLTestUnlabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Predicting on {len(test_ds)} test samples...")
        test_results = predict_test(model, test_loader, device=device)

        # Held-out good predictions (for ensemble weight-fitter; same paths as
        # the other models via the shared SEED'd split in make_labeled_eval_from_train).
        good_eval_loader = DataLoader(val_good_subset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Predicting on {len(val_good_subset)} held-out good samples...")
        good_results = predict_test(model, good_eval_loader, device=device)

        # Dump raw maps BEFORE sample gate so ensemble controls gating.
        if args.dump_raw:
            dump_labeled(args.dump_raw, args.model_name, class_name, results)
            dump_test(args.dump_raw, args.model_name, class_name, test_results)
            dump_good_eval(args.dump_raw, args.model_name, class_name, good_results)
        if args.view_gate_alpha > 0:
            apply_sample_gate(test_results, alpha=args.view_gate_alpha)
        submission_rows.extend(encode_submission_rows(test_results))

    metrics_path = P(args.output_dir) / "metrics.csv"
    if all_metrics:
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    mean_ap = float(np.mean([m['pixel_ap'] for m in all_metrics])) if all_metrics else float('nan')
    mean_auroc = float(np.mean([m['pixel_auroc'] for m in all_metrics])) if all_metrics else float('nan')
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Mean Pixel AP (train-set eval, optimistic): {mean_ap:.4f}")
    print(f"Mean Pixel AUROC (train-set eval, optimistic): {mean_auroc:.4f}")

    pdf_path = P(args.output_dir) / "predictions.pdf"
    save_results_to_pdf(pdf_results, str(pdf_path))
    print(f"Saved PDF to {pdf_path}")

    sub_dir = P("submission")
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / 'submission_simplenet_v2.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Label'])
        writer.writeheader()
        writer.writerows(submission_rows)
    zip_path = sub_dir / 'submission_simplenet_v2.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(csv_path, arcname=csv_path.name)
    print(f"Saved submission to {zip_path}")


if __name__ == "__main__":
    main()
