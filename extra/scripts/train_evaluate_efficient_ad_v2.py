import sys
sys.path.append('.')
import argparse
import os
import zipfile
from itertools import cycle
from pathlib import Path as P

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
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
from adl_lib.efficient_ad import EfficientAD, DEFAULT_TEACHER_CKPT
from adl_lib.ensemble import apply_sample_gate
from adl_lib.dump import dump_labeled, dump_test, dump_good_eval


# ---------- penalty (OOD) loader ----------

def build_penalty_loader(image_size, batch_size, num_workers, pin_memory, root="data/stl10"):
    """STL10 train split (~13MB, 5k natural images) as OOD stream for EfficientAD's
    penalty loss. Returns tensors in [0, 1] to match ADL datasets."""
    os.makedirs(root, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # already in [0, 1]
    ])
    ds = STL10(root=root, split="train", download=True, transform=tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )


# ---------- losses ----------

def top_k_student_loss(teacher_features, student_features, q=0.1):
    """Hard-pixel (top q-fraction hardest pixels) student loss. Paper uses q=0.1."""
    diff = (teacher_features - student_features) ** 2  # [B, C, H, W]
    flat = diff.flatten(1)  # [B, C*H*W]
    k = max(1, int(flat.shape[1] * q))
    topk, _ = torch.topk(flat, k, dim=1)
    return topk.mean()


def margin_mask_loss(anomaly_map, mask, margin=0.1):
    """Ranking-style aux loss: mean score inside mask should exceed mean score outside.

    Works for any non-negative anomaly map and for unbounded logits.
    """
    pos_sum = (anomaly_map * mask).sum(dim=(1, 2, 3))
    neg_sum = (anomaly_map * (1 - mask)).sum(dim=(1, 2, 3))
    pos_cnt = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    neg_cnt = (1 - mask).sum(dim=(1, 2, 3)).clamp(min=1.0)
    pos = pos_sum / pos_cnt
    neg = neg_sum / neg_cnt
    return F.relu(margin - (pos - neg)).mean()


# ---------- training ----------

def train_efficient_ad(
    model, train_loader, penalty_loader, anom_loader,
    epochs=15, lr=1e-4, device="cuda",
    student_topk=0.1, w_aux=0.05,
):
    model.to(device)
    optimizer = Adam(
        list(model.student.parameters()) + list(model.autoencoder.parameters()),
        lr=lr,
    )

    pen_iter = cycle(penalty_loader)
    anom_iter = cycle(anom_loader) if anom_loader is not None else None

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            teacher_features, student_features, ae_features = model(images)

            loss_st = top_k_student_loss(teacher_features, student_features, q=student_topk)
            loss_ae = F.mse_loss(ae_features, teacher_features.detach())

            # Penalty: student should NOT mimic teacher on OOD natural images.
            pen_batch, _ = next(pen_iter)
            pen_images = pen_batch.to(device)
            s_pen = model.student(model._preprocess(pen_images))
            loss_pen = (s_pen ** 2).mean()

            loss = loss_st + loss_pen + loss_ae

            # Auxiliary supervised loss on labeled anomalies (small weight).
            if anom_iter is not None and w_aux > 0:
                aux_batch = next(anom_iter)
                aux_images = aux_batch["image"].to(device)
                aux_masks = aux_batch["mask"].to(device)  # [B, 1, H, W] in {0, 1}
                t_aux = model._teacher_features(aux_images, update_stats=False)
                s_aux = model.student(model._preprocess(aux_images))
                a_aux = model.autoencoder(model._preprocess(aux_images))
                st_d = torch.mean((t_aux - s_aux) ** 2, dim=1, keepdim=True)
                ae_d = torch.mean((t_aux - a_aux) ** 2, dim=1, keepdim=True)
                amap = F.interpolate(st_d + ae_d, size=aux_masks.shape[-2:],
                                     mode="bilinear", align_corners=False)
                loss = loss + w_aux * margin_mask_loss(amap, aux_masks)

            loss.backward()
            optimizer.step()
            pbar.set_postfix({"st": f"{loss_st.item():.4f}",
                              "ae": f"{loss_ae.item():.4f}",
                              "pen": f"{loss_pen.item():.4f}"})
    return model


def evaluate_efficient_ad(model, val_loader, device="cuda"):
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
    """Per-class robust percentile clipping → [0, 1] → q8rle.

    Preserves dynamic range under 8-bit quantization (matters for ranking metrics)
    by ignoring extreme outliers when fixing the scale.
    """
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
    parser = argparse.ArgumentParser(description="Train and evaluate EfficientAD on all classes.")
    parser.add_argument("--classes", type=str, nargs="+", help="Classes to process. Defaults to all.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_CKPT)
    parser.add_argument("--student_topk", type=float, default=0.1)
    parser.add_argument("--w_aux", type=float, default=0.05,
                        help="Weight on auxiliary supervised mask loss; 0 disables.")
    parser.add_argument("--view_gate_alpha", type=float, default=0.5,
                        help="Multiplier strength for the sample-level multi-view gate; 0 disables.")
    parser.add_argument("--output_dir", type=str, default="artifacts/efficient_ad_v2")
    parser.add_argument("--dump_raw", type=str, default="artifacts/raw_outputs",
                        help="Root dir for raw anomaly-map dumps (for ensembling). "
                             "Empty string disables dumping.")
    parser.add_argument("--model_name", type=str, default="efficient_ad",
                        help="Sub-directory name under --dump_raw (must match what "
                             "the ensemble script expects).")
    args = parser.parse_args()

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.classes:
        classes = args.classes
    else:
        classes = sorted([d.name for d in P(PATH).iterdir() if d.is_dir() and d.name.startswith("class_")])
    print(f"Processing {len(classes)} classes: {classes}")

    # Shared OOD loader (downloads STL10 once).
    penalty_loader = build_penalty_loader(IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    all_metrics, pdf_results, submission_rows = [], [], []

    for class_name in classes:
        print(f"\n>>> Class: {class_name}")
        model = EfficientAD(teacher_ckpt=args.teacher_ckpt).to(device)

        # Held-out good split: train on train_good_subset; val_good_subset feeds the ensemble.
        train_good_subset, val_good_subset, _ = make_labeled_eval_from_train(
            PATH, class_name, image_size=IMAGE_SIZE,
            val_good_ratio=0.15, seed=SEED,
        )
        good_loader = DataLoader(train_good_subset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        anom_ds = ADLTrainAnomalyLabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        # Smaller batch since anom set is tiny; shuffle so different examples per epoch.
        anom_loader_train = DataLoader(
            anom_ds, batch_size=min(BATCH_SIZE, max(2, len(anom_ds))),
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        print(f"Training on {len(good_ds)} good + {len(anom_ds)} labeled anomaly samples...")
        train_efficient_ad(
            model, good_loader, penalty_loader,
            anom_loader=anom_loader_train if args.w_aux > 0 else None,
            epochs=args.epochs, lr=args.lr, device=device,
            student_topk=args.student_topk, w_aux=args.w_aux,
        )

        # Calibrate per-stream quantiles on normal images (paper's q_a/q_b).
        print("Calibrating per-stream map quantiles on normal images...")
        model.compute_map_quantiles(good_loader, device=device, q_a=0.9, q_b=0.995)

        # Local validation reusing the same labeled anomalies — optimistic, but
        # we just want PDFs and a sanity number. Real validation is Kaggle.
        val_loader = DataLoader(anom_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Evaluating on {len(anom_ds)} labeled anomaly samples (NOTE: training-set overlap)...")
        results = evaluate_efficient_ad(model, val_loader, device=device)

        metrics = summarize_metrics(results)
        metrics['class'] = class_name
        all_metrics.append(metrics)
        print(f"[train-set eval] Pixel AP: {metrics['pixel_ap']:.4f} | Pixel AUROC: {metrics['pixel_auroc']:.4f}")

        thr = metrics.get('pixel_threshold', 0.5)
        if np.isnan(thr):
            thr = 0.5
        pdf_results.append((class_name, results, thr))

        test_ds = ADLTestUnlabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Predicting on {len(test_ds)} test samples...")
        test_results = predict_test(model, test_loader, device=device)

        # Held-out good predictions for ensemble weight-fitting.
        good_eval_loader = DataLoader(val_good_subset, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Predicting on {len(val_good_subset)} held-out good samples...")
        good_results = predict_test(model, good_eval_loader, device=device)

        # Dump raw maps BEFORE the sample gate so the ensemble step controls gating.
        if args.dump_raw:
            dump_labeled(args.dump_raw, args.model_name, class_name, results)
            dump_test(args.dump_raw, args.model_name, class_name, test_results)
            dump_good_eval(args.dump_raw, args.model_name, class_name, good_results)
        if args.view_gate_alpha > 0:
            apply_sample_gate(test_results, alpha=args.view_gate_alpha)
        submission_rows.extend(encode_submission_rows(test_results))

    # Save Metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = P(args.output_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Mean Pixel AP (train-set eval, optimistic): {metrics_df['pixel_ap'].mean():.4f}")
    print(f"Mean Pixel AUROC (train-set eval, optimistic): {metrics_df['pixel_auroc'].mean():.4f}")

    # PDF
    pdf_path = P(args.output_dir) / "predictions.pdf"
    save_results_to_pdf(pdf_results, str(pdf_path))
    print(f"Saved PDF to {pdf_path}")

    # Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_dir = P("submission")
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / 'submission_efficient_ad_v2.csv'
    sub_df.to_csv(csv_path, index=False)
    zip_path = sub_dir / 'submission_efficient_ad_v2.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(csv_path, arcname=csv_path.name)
    print(f"Saved submission to {zip_path}")


if __name__ == "__main__":
    main()
