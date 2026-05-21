import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# Suppress annoying scikit-learn warnings when no positive samples are present
warnings.filterwarnings("ignore", message="No positive class found in y_true")

from scipy.ndimage import label, binary_dilation

from adl_lib.config import (
    BATCH_SIZE,
    DEVICE,
    EXPORT_DPI,
    EXPORT_FORCE_REBUILD,
    EXPORT_MAX_IMAGES_PER_CLASS,
    EXPORT_PDF_PATH,
    HEATMAP_OVERLAY_ALPHA,
    IMAGE_SIZE,
    NUM_WORKERS,
    PATH,
    PIN_MEMORY,
    SCORE_DISTRIBUTION_BINS,
    SEED,
    SHOW_DETECTION_N,
)
from adl_lib.data import tensor_to_numpy_image, tensor_to_numpy_mask


def _gaussian_kernel(
    window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32
):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def _filter2d(x, kernel):
    pad = kernel.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=x.shape[1])


def gaussian_blur(x, sigma=4.0):
    if sigma is None or sigma <= 0:
        return x
    kernel_size = int(max(3, 2 * round(4 * sigma) + 1))
    kernel_size += 1 - (kernel_size % 2)
    kernel = _gaussian_kernel(
        window_size=kernel_size,
        sigma=float(sigma),
        channels=x.shape[1],
        device=x.device,
        dtype=x.dtype,
    )
    return _filter2d(x, kernel)


def remove_dust_components(
    amap, threshold_percentile=75.0, min_area_ratio=0.2, dilation_iters=25
):
    """
    Identifies 'core' connected components at a moderate threshold, 
    removes those that are significantly smaller than the largest core,
    and dilates the surviving cores to preserve the continuous scores ('halo').
    """
    # 1. Binarize based on percentile to find 'cores'
    thresh = np.percentile(amap, threshold_percentile)
    if thresh <= 0:
        thresh = amap.max() * 0.5  # fallback if map is mostly zero

    core_mask = (amap > thresh).astype(np.uint8)

    # 2. Label core components
    labeled_cores, n_components = label(core_mask)
    if n_components == 0:
        return np.zeros_like(amap)

    # 3. Calculate areas and find largest core
    component_ids, counts = np.unique(labeled_cores[labeled_cores > 0], return_counts=True)
    max_area = counts.max()

    # 4. Filter cores by area ratio
    keep_ids = component_ids[counts >= (max_area * min_area_ratio)]

    # 5. Create mask of kept cores
    cleaned_core_mask = np.isin(labeled_cores, keep_ids)

    # 6. Dilate the kept cores to include the surrounding continuous scores
    dilated_mask = binary_dilation(cleaned_core_mask, iterations=dilation_iters).astype(
        np.float32
    )

    return amap * dilated_mask


def postprocess_anomaly_map(anomaly_map, sigma=2.0, background_percentile=10.0):
    amap = np.asarray(anomaly_map, dtype=np.float32)
    amap_t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    smoothed = gaussian_blur(amap_t, sigma=float(sigma))[0, 0].detach().cpu().numpy()
    
    # We removed the aggressive floor clipping and dust removal,
    # as these can destroy valid anomalies and complicate global normalization.
    return smoothed


def calibrate_threshold_from_labeled(model, labeled_loader):
    all_scores = []
    all_labels = []

    results = model.predict_labeled(labeled_loader)

    for r in results:
        if int(r.get("label", 0)) != 1:
            continue

        amap = np.asarray(r["anomaly_map"], dtype=np.float32)
        mask = (np.asarray(r["mask"]) > 0).astype(np.uint8)

        all_scores.append(amap.reshape(-1))
        all_labels.append(mask.reshape(-1))

    if len(all_scores) == 0:
        print(
            "No labeled anomaly samples found for calibration; using default threshold 0.0"
        )
        return 0.0, np.nan

    all_scores = np.concatenate(all_scores).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.uint8)

    pixel_ap = average_precision_score(all_labels, all_scores)
    print(f"Pixel AP on labeled anomaly set: {pixel_ap:.4f}")

    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    if thresholds.size == 0:
        best_threshold = float(np.median(all_scores))
        best_f1 = 0.0
    else:
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])

    print(f"Best pixel threshold: {best_threshold:.4f}  (F1={best_f1:.4f})")
    return best_threshold, float(pixel_ap)


def normalize_map(amap):
    amap = np.asarray(amap, dtype=np.float32)
    if amap.max() <= amap.min():
        return np.zeros_like(amap)
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)


def safe_metric(metric_fn, y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(metric_fn(y_true, scores))


def youden_threshold(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def summarize_metrics(results):
    image_labels = np.array([r["label"] for r in results], dtype=np.int64)
    image_scores = np.array([r["score"] for r in results], dtype=np.float32)
    pixel_masks = np.concatenate([r["mask"].reshape(-1) for r in results]).astype(
        np.uint8
    )
    pixel_scores = np.concatenate(
        [r["anomaly_map"].reshape(-1) for r in results]
    ).astype(np.float32)

    return {
        "image_auroc": safe_metric(roc_auc_score, image_labels, image_scores),
        "image_ap": safe_metric(average_precision_score, image_labels, image_scores),
        "image_threshold": youden_threshold(image_labels, image_scores),
        "pixel_auroc": safe_metric(roc_auc_score, pixel_masks, pixel_scores),
        "pixel_ap": safe_metric(average_precision_score, pixel_masks, pixel_scores),
        "pixel_threshold": youden_threshold(pixel_masks, pixel_scores),
    }


def heatmap_overlay(image, amap, alpha=None):
    if alpha is None:
        alpha = HEATMAP_OVERLAY_ALPHA
    image = tensor_to_numpy_image(image)
    heat = plt.cm.jet(normalize_map(amap))[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 1.0)


def plot_score_distribution_v2(results, title="Image-level anomaly scores", bins=None):
    if bins is None:
        bins = SCORE_DISTRIBUTION_BINS
    normal_scores = [r["score"] for r in results if r.get("label", 0) == 0]
    anomaly_scores = [r["score"] for r in results if r.get("label", 0) == 1]

    all_scores = normal_scores + anomaly_scores
    if len(all_scores) == 0:
        return

    bins = np.linspace(min(all_scores), max(all_scores), bins)
    plt.figure(figsize=(7, 4))
    plt.hist(normal_scores, bins=bins, density=True, alpha=0.5, label="normal")
    plt.hist(anomaly_scores, bins=bins, density=True, alpha=0.5, label="anomalous")
    plt.xlabel("image anomaly score")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.show()


def show_detection_results(results, n=None, thresholds=None, title=None):
    if n is None:
        n = SHOW_DETECTION_N
    if len(results) == 0:
        return

    order = np.argsort([r["score"] for r in results])[::-1]
    chosen = order[: min(n, len(results))]

    fig, axes = plt.subplots(len(chosen), 4, figsize=(16, 3.8 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    for row, idx in enumerate(chosen):
        r = results[idx]

        img = r["image"]
        amap = r["anomaly_map"]
        mask = r["mask"]

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Input\\nlabel={r['label']} score={r['score']:.4f}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap_overlay(img, amap))
        axes[row, 1].set_title("Overlay")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(amap, cmap="jet")
        axes[row, 2].set_title("Anomaly map")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        if thresholds is None:
            pred = np.zeros_like(mask)
            thr_txt = "-"
        else:
            pred = (amap >= thresholds["pixel_threshold"]).astype(np.float32)
            thr_txt = f"{thresholds['pixel_threshold']:.4f}"

        axes[row, 3].imshow(mask, cmap="gray", alpha=0.8)
        axes[row, 3].imshow(pred, cmap="Reds", alpha=0.35)
        axes[row, 3].set_title(f"GT(gray) + Pred(red)\\nthr={thr_txt}")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()


def draw_box(ax, bbox, color="lime", linewidth=2):
    x0, y0, x1, y1 = bbox
    rect = plt.Rectangle(
        (x0, y0),
        max(1, x1 - x0),
        max(1, y1 - y0),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)


def grid_coord_to_bbox(coord, image_shape, grid_shape):
    gy, gx = int(coord[0]), int(coord[1])
    h_img, w_img = image_shape[:2]
    h_grid, w_grid = grid_shape

    y0 = int(round(gy * h_img / h_grid))
    y1 = int(round((gy + 1) * h_img / h_grid))
    x0 = int(round(gx * w_img / w_grid))
    x1 = int(round((gx + 1) * w_img / w_grid))

    y0 = max(0, min(y0, h_img - 1))
    x0 = max(0, min(x0, w_img - 1))
    y1 = max(y0 + 1, min(y1, h_img))
    x1 = max(x0 + 1, min(x1, w_img))
    return x0, y0, x1, y1


def crop_with_bbox(image, bbox):
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def explain_patchcore_match(patchcore, results, rank=0):
    if len(results) == 0:
        print("No results available.")
        return

    if not patchcore.enable_image_cache or len(patchcore.train_images) == 0:
        print(
            "Training images are not cached. Enable ENABLE_IMAGE_CACHE=True to use explainability."
        )
        return

    order = np.argsort([r["score"] for r in results])[::-1]
    rank = int(np.clip(rank, 0, len(order) - 1))
    r = results[order[rank]]

    patch_scores = r["patch_scores_small"]
    y, x = np.unravel_index(np.argmax(patch_scores), patch_scores.shape)

    bank_idx = int(r["nn_indices_small"][y, x])
    source_img_id = int(patchcore.bank_image_ids[bank_idx].item())
    source_coord = patchcore.bank_coords[bank_idx].tolist()

    test_img = r["image"]
    train_img = patchcore.train_images[source_img_id]
    if train_img is None:
        print("Training image not cached for this index.")
        return

    train_img = train_img.astype(np.float32) / 255.0

    test_bbox = grid_coord_to_bbox((y, x), test_img.shape, patch_scores.shape)
    train_bbox = grid_coord_to_bbox(
        source_coord, train_img.shape, patchcore.feature_grid_shape
    )

    test_patch = crop_with_bbox(test_img, test_bbox)
    train_patch = crop_with_bbox(train_img, train_bbox)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(test_img)
    axes[0].imshow(normalize_map(r["anomaly_map"]), cmap="jet", alpha=0.35)
    draw_box(axes[0], test_bbox)
    axes[0].set_title(f"Query image\\nscore={r['score']:.4f}")
    axes[0].axis("off")

    axes[1].imshow(test_patch)
    axes[1].set_title("Most anomalous query patch")
    axes[1].axis("off")

    axes[2].imshow(train_img)
    draw_box(axes[2], train_bbox)
    axes[2].set_title("Nearest normal training image")
    axes[2].axis("off")

    axes[3].imshow(train_patch)
    axes[3].set_title("Nearest normal patch")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    print("Query image:", r["path"])
    print("Nearest normal image:", patchcore.train_paths[source_img_id])
    print("Query patch grid coordinate:", (int(y), int(x)))
    print("Nearest normal patch grid coordinate:", tuple(int(v) for v in source_coord))


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{prefix} GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"{prefix} Using CPU (no GPU memory tracking available)")


# ===== PDF EXPORT HELPERS =====


def classes_with_ground_truth(root):
    """Find all classes that have both ground truth masks and anomaly training data."""
    root = Path(root)
    classes = []
    for p in sorted(root.iterdir()):
        if not (p.is_dir() and p.name.startswith("class_")):
            continue
        gt_root = p / "ground_truth_train"
        train_root = p / "train"
        if gt_root.exists() and train_root.exists():
            has_gt = any(
                d.is_dir() and d.name.startswith("anomaly_") for d in gt_root.iterdir()
            )
            has_anom = any(
                d.is_dir() and d.name.startswith("anomaly_")
                for d in train_root.iterdir()
            )
            if has_gt and has_anom:
                classes.append(p.name)
    return classes


def build_patchcore_for_class(
    cls_name,
    patchcore_backbone_candidates,
    patchcore_out_indices,
    patchcore_patchsize,
    patchcore_coreset_fraction,
    patchcore_candidate_pool_size,
    patchcore_random_projection_dim,
    patchcore_bank_chunk_size,
    anomaly_map_sigma,
    anomaly_map_background_percentile,
    use_faiss,
):
    """Build or load PatchCore model for a specific class."""
    from adl_lib.patchcore import PatchCoreLite
    from adl_lib.data import make_labeled_eval_from_train

    pc = PatchCoreLite(
        backbone_candidates=patchcore_backbone_candidates,
        out_indices=patchcore_out_indices,
        patchsize=patchcore_patchsize,
        coreset_fraction=patchcore_coreset_fraction,
        candidate_pool_size=patchcore_candidate_pool_size,
        projection_dim=patchcore_random_projection_dim,
        bank_chunk_size=patchcore_bank_chunk_size,
        sigma=anomaly_map_sigma,
        background_percentile=anomaly_map_background_percentile,
        enable_image_cache=False,
        use_faiss=use_faiss,
    )
    pc.class_name = cls_name

    # Fast path: if FAISS index exists, load it and skip re-fit
    loaded = None
    if (not EXPORT_FORCE_REBUILD) and pc.use_faiss:
        loaded = pc._load_faiss_index(cls_name, use_gpu=True)

    if loaded is None:
        train_good_subset, _, _ = make_labeled_eval_from_train(
            PATH,
            cls_name,
            image_size=IMAGE_SIZE,
            val_good_ratio=0.0,
            seed=SEED,
        )
        fit_loader = DataLoader(
            train_good_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        pc.fit(fit_loader, class_name=cls_name)

    return pc


def predict_labeled_anomaly_results(pc, cls_name):
    """Predict on labeled anomaly data with class-wise threshold calibration."""
    from adl_lib.data import ADLTrainAnomalyLabeledDataset

    labeled_anom_ds = ADLTrainAnomalyLabeledDataset(
        PATH, cls_name, image_size=IMAGE_SIZE
    )
    if EXPORT_MAX_IMAGES_PER_CLASS is not None:
        n_keep = min(int(EXPORT_MAX_IMAGES_PER_CLASS), len(labeled_anom_ds))
        labeled_anom_ds = Subset(labeled_anom_ds, list(range(n_keep)))

    loader = DataLoader(
        labeled_anom_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Class-wise threshold calibration on the same labeled anomaly set.
    best_thr, _ = calibrate_threshold_from_labeled(pc, loader)
    results = pc.predict_labeled(loader)
    return results, float(best_thr)


def save_results_to_pdf(all_class_results, output_pdf):
    """Export per-image visualizations (Input | Anomaly map | GT + Pred overlay) to PDF."""
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    with PdfPages(output_pdf) as pdf:
        for cls_name, class_results, thr in all_class_results:
            for i, r in enumerate(class_results, start=1):
                n_total += 1
                img = r["image"]
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0.0, 1.0)
                
                amap = r["anomaly_map"]
                gt = (np.asarray(r["mask"]) > 0).astype(np.float32).squeeze()
                pred = (np.asarray(amap) >= thr).astype(np.float32).squeeze()

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

                axes[0].imshow(img)
                axes[0].set_title("Input")
                axes[0].axis("off")

                im = axes[1].imshow(amap, cmap="jet")
                axes[1].set_title(f"Anomaly map\nscore={r['score']:.4f}")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(gt, cmap="gray", alpha=0.85)
                axes[2].imshow(pred, cmap="Reds", alpha=0.35)
                axes[2].set_title(f"GT(gray) + Pred(red)\nthr={thr:.4f}")
                axes[2].axis("off")

                fig.suptitle(
                    f"{cls_name} | {i}/{len(class_results)} | {Path(r['path']).name}",
                    fontsize=11,
                    y=1.02,
                )
                fig.tight_layout()
                pdf.savefig(fig, dpi=EXPORT_DPI, bbox_inches="tight")
                plt.close(fig)

    return n_total


def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{prefix} GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"{prefix} Using CPU (no GPU memory tracking available)")
