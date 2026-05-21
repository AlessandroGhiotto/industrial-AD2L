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
from torchvision.datasets import DTD
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
from adl_lib.draem import DRAEM, DraemAnomalyGenerator, FocalLoss
from adl_lib.ensemble import apply_sample_gate
from adl_lib.dump import dump_labeled, dump_test, dump_good_eval


# ---------- texture (anomaly source) loader ----------

def build_texture_loader(image_size, batch_size, num_workers, pin_memory, root="data/dtd"):
    """Describable Textures Dataset as the anomaly-source image stream.

    Auto-downloads on first run (~600MB). Each batch is a [B, 3, H, W] tensor
    in [0, 1] matching the ADL image format.
    """
    os.makedirs(root, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    ds = DTD(root=root, split="train", download=True, transform=tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )


# ---------- aux loss ----------

def margin_mask_loss(score_map, mask, margin=0.1):
    """Inside-mask scores should exceed outside-mask scores by `margin`."""
    pos_sum = (score_map * mask).sum(dim=(1, 2, 3))
    neg_sum = (score_map * (1 - mask)).sum(dim=(1, 2, 3))
    pos_cnt = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    neg_cnt = (1 - mask).sum(dim=(1, 2, 3)).clamp(min=1.0)
    return F.relu(margin - (pos_sum / pos_cnt - neg_sum / neg_cnt)).mean()


# ---------- training ----------

def train_draem(
    model, train_loader, texture_loader, anom_loader,
    epochs=15, lr=1e-4, device="cuda",
    focal_gamma=2.0, p_aug=0.5, w_aux=0.05, aux_margin=0.1,
    amp=True, accum_steps=1,
):
    model.to(device)
    optimizer = Adam(list(model.parameters()), lr=lr)
    gen = DraemAnomalyGenerator(p_aug=p_aug, seed=SEED)
    focal = FocalLoss(gamma=focal_gamma)

    use_amp = bool(amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tex_iter = cycle(texture_loader)
    anom_iter = cycle(anom_loader) if anom_loader is not None else None

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            B = images.shape[0]

            # Gather B textures; DTD batches may be smaller than B near epoch boundaries.
            tex_chunks = []
            collected = 0
            while collected < B:
                tex_batch, _ = next(tex_iter)
                tex_chunks.append(tex_batch)
                collected += tex_batch.shape[0]
            textures = torch.cat(tex_chunks, dim=0)[:B].to(device, non_blocking=True)

            aug_images, aug_masks = gen(images, textures)  # [B,3,H,W], [B,1,H,W]

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, logits = model(aug_images)
                loss_rec = F.mse_loss(recon, images)
                loss_seg = focal(logits, aug_masks.squeeze(1))
                loss = loss_rec + loss_seg

                if anom_iter is not None and w_aux > 0:
                    aux_batch = next(anom_iter)
                    aux_images = aux_batch["image"].to(device, non_blocking=True)
                    aux_masks = aux_batch["mask"].to(device, non_blocking=True)
                    prob = model.anomaly_prob(aux_images)
                    loss = loss + w_aux * margin_mask_loss(prob, aux_masks, margin=aux_margin)

                loss = loss / accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({"rec": f"{loss_rec.item():.4f}",
                              "seg": f"{loss_seg.item():.4f}"})
    return model


def evaluate_draem(model, val_loader, device="cuda"):
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
    """Per-class percentile clipping -> [0, 1] -> q8rle (preserves dynamic range)."""
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
    parser = argparse.ArgumentParser(description="Train and evaluate DRAEM on all classes.")
    parser.add_argument("--classes", type=str, nargs="+")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32,
                        help="U-Net width; 32 fits ~batch 16 in 12GB with AMP. "
                             "Bump to 64 only if you have >=24GB.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Overrides config.BATCH_SIZE for this script.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Mixed-precision training (default on).")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient-accumulation steps to keep effective batch large "
                             "while shrinking the per-step batch.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--p_aug", type=float, default=0.5,
                        help="Probability of applying synthetic anomaly to each training image.")
    parser.add_argument("--w_aux", type=float, default=0.05,
                        help="Weight on auxiliary supervised mask loss; 0 disables.")
    parser.add_argument("--aux_margin", type=float, default=0.1)
    parser.add_argument("--view_gate_alpha", type=float, default=0.5,
                        help="Multiplier strength for the sample-level multi-view gate; 0 disables.")
    parser.add_argument("--output_dir", type=str, default="artifacts/draem_v2")
    parser.add_argument("--dump_raw", type=str, default="artifacts/raw_outputs",
                        help="Root dir for raw anomaly-map dumps (for ensembling). "
                             "Empty string disables dumping.")
    parser.add_argument("--model_name", type=str, default="draem",
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
    print(f"Processing {len(classes)} classes with DRAEM")

    batch_size = args.batch_size
    print(f"Using batch_size={batch_size}, base_channels={args.base_channels}, "
          f"amp={args.amp}, accum_steps={args.accum_steps}")

    # Shared texture loader (downloads DTD once, ~600MB).
    texture_loader = build_texture_loader(IMAGE_SIZE, batch_size, NUM_WORKERS, PIN_MEMORY)

    all_metrics, pdf_results, submission_rows = [], [], []

    for class_name in classes:
        print(f"\n>>> Class: {class_name}")
        model = DRAEM(base=args.base_channels).to(device)

        # Held-out good split: train on train_good_subset, evaluate good-eval on val_good_subset.
        train_good_subset, val_good_subset, _ = make_labeled_eval_from_train(
            PATH, class_name, image_size=IMAGE_SIZE,
            val_good_ratio=0.15, seed=SEED,
        )
        good_loader = DataLoader(train_good_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        anom_ds = ADLTrainAnomalyLabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        anom_loader_train = DataLoader(
            anom_ds, batch_size=min(batch_size, max(2, len(anom_ds))),
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        print(f"Training on {len(good_ds)} good + {len(anom_ds)} labeled anomaly samples...")
        train_draem(
            model, good_loader, texture_loader,
            anom_loader=anom_loader_train if args.w_aux > 0 else None,
            epochs=args.epochs, lr=args.lr, device=device,
            focal_gamma=args.focal_gamma, p_aug=args.p_aug,
            w_aux=args.w_aux, aux_margin=args.aux_margin,
            amp=args.amp, accum_steps=args.accum_steps,
        )

        val_loader = DataLoader(anom_ds, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Evaluating on {len(anom_ds)} labeled anomaly samples (NOTE: training-set overlap)...")
        results = evaluate_draem(model, val_loader, device=device)

        metrics = summarize_metrics(results)
        metrics['class'] = class_name
        all_metrics.append(metrics)
        print(f"[train-set eval] Pixel AP: {metrics['pixel_ap']:.4f} | Pixel AUROC: {metrics['pixel_auroc']:.4f}")

        thr = metrics.get('pixel_threshold', 0.5)
        if np.isnan(thr):
            thr = 0.5
        pdf_results.append((class_name, results, thr))

        test_ds = ADLTestUnlabeledDataset(PATH, class_name, image_size=IMAGE_SIZE)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        print(f"Predicting on {len(test_ds)} test samples...")
        test_results = predict_test(model, test_loader, device=device)

        # Held-out good predictions for the ensemble weight-fitter.
        good_eval_loader = DataLoader(val_good_subset, batch_size=batch_size, shuffle=False,
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

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = P(args.output_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Mean Pixel AP (train-set eval, optimistic): {metrics_df['pixel_ap'].mean():.4f}")
    print(f"Mean Pixel AUROC (train-set eval, optimistic): {metrics_df['pixel_auroc'].mean():.4f}")

    pdf_path = P(args.output_dir) / "predictions.pdf"
    save_results_to_pdf(pdf_results, str(pdf_path))
    print(f"Saved PDF to {pdf_path}")

    sub_df = pd.DataFrame(submission_rows)
    sub_dir = P("submission")
    sub_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sub_dir / 'submission_draem_v2.csv'
    sub_df.to_csv(csv_path, index=False)
    zip_path = sub_dir / 'submission_draem_v2.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(csv_path, arcname=csv_path.name)
    print(f"Saved submission to {zip_path}")


if __name__ == "__main__":
    main()
