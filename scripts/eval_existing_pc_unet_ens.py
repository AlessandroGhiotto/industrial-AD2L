"""Compute pooled pixel-AP for the 3 already-trained methods: PatchCore, U-Net,
and the Ensemble — for every class.

This mirrors the pipeline in `final/Ensemble.ipynb` (the same `_raw_maps`,
`_pc_signal`, p99 calibration and per-class alpha grid search) but evaluates
PC and U-Net independently as well as the fused map, so all three rows of
Table 1 fall out of one pass.

Requires:
  - PatchCore bank cache under `cfg.PC_BANKS_DIR / Lxx_yy_pcaDDD/`
    (built by `final/PatchCore.ipynb`).
  - One U-Net checkpoint per class at `cfg.UNET_MODEL_DIR / class_XX / best.pt`.

Writes:
  - outputs/table_results/patchcore.json
  - outputs/table_results/unet.json
  - outputs/table_results/ensemble.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
sys.path.insert(0, str(FINAL_DIR))

from src import config_final as cfg  # noqa: E402
from src import ensemble_final as ens  # noqa: E402
from src import patchcore_final as pc  # noqa: E402
from src import unet_final as unet  # noqa: E402
from src.eval_table import (  # noqa: E402
    _list_labeled_anomalies,
    pooled_pixel_ap_for_class,
    save_results,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GOOD_CALIB = 32  # images used to calibrate p99 per class
ALPHA_GRID = cfg.ENSEMBLE_ALPHAS

OUT_DIR = cfg.OUTPUT_DIR / "table_results"


# ── PatchCore bank loading (copied from PatchCore.ipynb LOAD_BANKS branch) ────
def load_pc_artifacts(classes: list[str]):
    extractor = pc.DINOv2(
        cfg.PC_BACKBONE, cfg.PC_LAYERS, cfg.PC_GRID_SIZE, DEVICE
    ).to(DEVICE).eval()
    # ensemble_final.infer_pc_map expects the PER-LAYER embed dim
    # (it internally multiplies by len(layers)). This auto-adapts to
    # whichever DINOv2 variant the cached bank was built with:
    #   base  → embed_dim=768   (combined feat = 4*768  = 3072)
    #   large → embed_dim=1024  (combined feat = 4*1024 = 4096)
    feat_dim = extractor.backbone.embed_dim
    layers_str = "_".join(map(str, cfg.PC_LAYERS))
    bank_dir = cfg.PC_BANKS_DIR / f"L{layers_str}_pca{cfg.PC_PCA_DIM}"
    if not bank_dir.exists():
        raise FileNotFoundError(
            f"PatchCore bank dir not found: {bank_dir}\n"
            f"Run final/PatchCore.ipynb (bank-build cell) first."
        )
    pca = pc.GPUPCA(n_components=cfg.PC_PCA_DIM, device=DEVICE)
    pca.mean_ = torch.from_numpy(np.load(bank_dir / "pca_mean.npy")).to(DEVICE)
    pca.components_ = torch.from_numpy(
        np.load(bank_dir / "pca_components.npy")
    ).to(DEVICE)
    banks: dict[str, torch.Tensor] = {}
    for cls in classes:
        arr = np.load(bank_dir / f"{cls}_bank.npy")
        banks[cls] = torch.from_numpy(arr).to(DEVICE)
    return extractor, pca, banks, feat_dim


# ── Raw map helpers (verbatim port of _raw_maps / _pc_signal from Ensemble.ipynb)
def raw_maps(unet_model, img_pil, img_path: Path, bank, extractor, pca, feat_dim,
             use_tta: bool):
    fg = unet.foreground_from_image(
        img_pil.resize((cfg.UNET_IMAGE_SIZE, cfg.UNET_IMAGE_SIZE))
    )[0].numpy()
    view_id = unet._parse_view_and_id(img_path.name)[1]
    v_t = torch.tensor([view_id], device=DEVICE, dtype=torch.long)
    u_raw = ens.infer_unet_map(unet, unet_model, img_pil, view_id, DEVICE,
                               cfg.UNET_IMAGE_SIZE, use_tta=use_tta, fg_mask=fg)
    p_raw = ens.infer_pc_map(pc, extractor, pca, bank, img_path, DEVICE,
                             cfg.PC_LAYERS, cfg.PC_GRID_SIZE,
                             cfg.UNET_IMAGE_SIZE, feat_dim=feat_dim,
                             fg_mask=None)
    p_clean = ens.clean_pc_map(p_raw, fg_mask=fg,
                               erode_px=cfg.PC_FG_ERODE_PX,
                               open_radius=cfg.PC_OPEN_RADIUS)
    return u_raw, p_clean, fg


def pc_signal(p_clean, fg, p_p99):
    p_ex = ens.excess_above_p99(p_clean, p_p99)
    p_ex = ens.remove_small_blobs(p_ex, cfg.PC_MIN_AREA, threshold=0.0)
    return p_ex * fg


# ── Per-class p99 calibration ─────────────────────────────────────────────────
def calibrate_pc_p99(cls: str, bank, extractor, pca, feat_dim,
                     unet_model) -> float:
    good_dir = cfg.DATASET_DIR / cls / "train" / "good"
    paths = sorted(good_dir.glob("*.png"))[:N_GOOD_CALIB]
    clean_maps = []
    for gp in paths:
        img_pil = Image.open(gp).convert("RGB")
        _, p_clean, _ = raw_maps(unet_model, img_pil, gp, bank,
                                 extractor, pca, feat_dim, use_tta=False)
        clean_maps.append(p_clean)
    _, p_p99 = ens.compute_calibration_stats(clean_maps)
    return float(p_p99)


# ── Per-class alpha optimization on labeled anomalies ─────────────────────────
def optimize_alpha(cls: str, bank, extractor, pca, feat_dim, unet_model,
                   p_p99) -> float:
    items = _list_labeled_anomalies(cfg.DATASET_DIR, cls)
    if not items:
        return 0.5
    u_val, p_val, gt = [], [], []
    for it in items:
        img_pil = Image.open(it["img"]).convert("RGB")
        u_raw, p_clean, fg = raw_maps(unet_model, img_pil, it["img"], bank,
                                      extractor, pca, feat_dim,
                                      use_tta=cfg.UNET_USE_TTA)
        u_val.append(u_raw)
        p_val.append(pc_signal(p_clean, fg, p_p99))
        gt_img = Image.open(it["mask"]).convert("L").resize(
            (u_raw.shape[1], u_raw.shape[0]), Image.NEAREST)
        gt.append((np.array(gt_img) > 0).astype(np.uint8))
    best_alpha, _ = ens.optimize_weights_per_class(
        cls, u_val, p_val, gt, ALPHA_GRID,
        strategy=cfg.ENSEMBLE_STRATEGY)
    return float(best_alpha)


# ── Per-class evaluation: PC, U-Net, Ensemble in one pass ─────────────────────
def evaluate_class(cls: str, bank, extractor, pca, feat_dim) -> dict[str, float]:
    print(f"\n=== {cls} ===")
    ckpt = cfg.UNET_MODEL_DIR / cls / "best.pt"
    unet_model = unet.load_model(str(ckpt), DEVICE)

    print("  calibrating PC p99...")
    p_p99 = calibrate_pc_p99(cls, bank, extractor, pca, feat_dim, unet_model)
    print(f"    p99 = {p_p99:.4f}")

    print("  optimizing alpha...")
    alpha = optimize_alpha(cls, bank, extractor, pca, feat_dim,
                           unet_model, p_p99)
    print(f"    alpha = {alpha:.2f}")

    print("  computing pooled pixel-AP (PC / U-Net / Ens)...")
    # We need three predict_fns; rather than re-running inference three times,
    # compute once per image and pool ourselves.
    items = _list_labeled_anomalies(cfg.DATASET_DIR, cls)
    pc_preds, u_preds, ens_preds, masks = [], [], [], []
    for it in items:
        img_pil = Image.open(it["img"]).convert("RGB")
        u_raw, p_clean, fg = raw_maps(unet_model, img_pil, it["img"], bank,
                                      extractor, pca, feat_dim,
                                      use_tta=cfg.UNET_USE_TTA)
        p_sig = pc_signal(p_clean, fg, p_p99)
        fused = ens.fuse_maps(u_raw, p_sig, alpha, strategy=cfg.ENSEMBLE_STRATEGY)
        gt_img = Image.open(it["mask"]).convert("L").resize(
            (u_raw.shape[1], u_raw.shape[0]), Image.NEAREST)
        m = (np.array(gt_img) > 0).astype(np.uint8)
        pc_preds.append(p_sig.ravel())
        u_preds.append(u_raw.ravel())
        ens_preds.append(fused.ravel())
        masks.append(m.ravel())

    from sklearn.metrics import average_precision_score
    y_true = np.concatenate(masks)
    out = {
        "pc": float(average_precision_score(y_true, np.concatenate(pc_preds))),
        "unet": float(average_precision_score(y_true, np.concatenate(u_preds))),
        "ens": float(average_precision_score(y_true, np.concatenate(ens_preds))),
        "alpha": alpha,
        "p99": p_p99,
    }
    print(f"    PC={out['pc']:.4f}  U-Net={out['unet']:.4f}  Ens={out['ens']:.4f}")
    return out


def main() -> None:
    classes = sorted(
        d.name for d in cfg.DATASET_DIR.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    )
    print(f"Classes: {classes}")
    extractor, pca, banks, feat_dim = load_pc_artifacts(classes)

    per_cls: dict[str, dict[str, float]] = {}
    for cls in classes:
        per_cls[cls] = evaluate_class(cls, banks[cls], extractor, pca, feat_dim)

    def collect(key: str) -> dict[str, float]:
        d = {cls: per_cls[cls][key] for cls in classes}
        d["mean"] = float(np.mean(list(d.values())))
        return d

    save_results(collect("pc"), OUT_DIR / "patchcore.json",
                 method_name=f"PatchCore ({cfg.PC_BACKBONE})")
    save_results(collect("unet"), OUT_DIR / "unet.json",
                 method_name="U-Net")
    save_results(collect("ens"), OUT_DIR / "ensemble.json",
                 method_name="Ensemble (PC + U-Net)")

    # Also dump the per-class alpha + p99 for reproducibility
    import json
    extra = {cls: {"alpha": per_cls[cls]["alpha"],
                   "p99": per_cls[cls]["p99"]} for cls in classes}
    (OUT_DIR / "ensemble_params.json").write_text(json.dumps(extra, indent=2))
    print(f"\nWrote per-class alpha/p99 → {OUT_DIR / 'ensemble_params.json'}")


if __name__ == "__main__":
    main()
