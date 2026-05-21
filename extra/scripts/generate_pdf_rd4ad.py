import sys
sys.path.append('.')
import torch
from pathlib import Path
from adl_lib.config import (
    PATH,
    RD4AD_BACKBONE,
    RD4AD_OUT_INDICES,
    RD4AD_BOTTLENECK_DIM,
    RD4AD_ANOMALY_MAP_SIGMA,
)
from adl_lib.rd4ad import RD4AD
from adl_lib.utils import (
    classes_with_ground_truth,
    predict_labeled_anomaly_results,
    save_results_to_pdf,
)

EXPORT_PDF_PATH_RD4AD = "./artifacts/rd4ad/all_classes_gt_predictions.pdf"

def build_rd4ad_for_class(cls_name):
    """Load RD4AD model for a specific class."""
    model = RD4AD(
        backbone_name=RD4AD_BACKBONE,
        out_indices=RD4AD_OUT_INDICES,
        bottleneck_dim=RD4AD_BOTTLENECK_DIM,
        sigma=RD4AD_ANOMALY_MAP_SIGMA,
    )
    loaded = model._load_checkpoint(cls_name)
    if not loaded:
        print(f"  Warning: No RD4AD checkpoint found for {cls_name}. Skipping.")
        return None
    return model

def main():
    print(f"Generating RD4AD PDF report using dataset: {PATH}")
    
    classes_gt = classes_with_ground_truth(PATH)
    if len(classes_gt) == 0:
        raise RuntimeError("No classes with ground truth were found.")

    print("Classes with ground truth:", classes_gt)
    all_results = []

    for cls_name in classes_gt:
        print(f"\n[Class] {cls_name}")
        
        model = build_rd4ad_for_class(cls_name)
        if model is None:
            continue
            
        class_results, class_thr = predict_labeled_anomaly_results(model, cls_name)
        print(f"  Labeled images: {len(class_results)} | threshold: {class_thr:.4f}")
        all_results.append((cls_name, class_results, class_thr))

    if not all_results:
        print("No results to export.")
        return

    n_saved = save_results_to_pdf(all_results, EXPORT_PDF_PATH_RD4AD)
    print(f"\nSaved PDF: {EXPORT_PDF_PATH_RD4AD}")
    print(f"Total rows/pages exported: {n_saved}")

if __name__ == "__main__":
    main()
