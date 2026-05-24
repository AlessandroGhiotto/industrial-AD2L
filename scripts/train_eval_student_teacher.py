"""Train + evaluate the Student-Teacher baseline on all 8 classes.

Uses the existing implementation in `final/extra/adl_lib/student_teacher.py`.
Hyperparameters mirror those in `final/extra/notebooks/student_teacher.ipynb`.
The result is one row of the Table 1 ablation: pooled pixel-AP per class
under the same protocol as the main models.

Checkpoints land in `outputs/student_teacher_models/`. JSON of per-class APs is
written to `outputs/table_results/student_teacher.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
# The student_teacher module exists in both `adl_lib/` (repo root) and
# `final/extra/adl_lib/` and the files are byte-identical. We use the
# repo-root version because its __init__.py imports from sibling modules
# (patchcore, birefnet, unet_synth) that only exist there.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FINAL_DIR))

from src import config_final as cfg  # noqa: E402
from src.eval_table import evaluate_all_classes, save_results  # noqa: E402

from adl_lib.student_teacher import (  # noqa: E402
    StudentTeacherAD,
    compute_student_teacher_maps,
    make_transforms,
    set_seed,
    train_student_teacher,
)


# Hyperparameters (copied from final/extra/notebooks/student_teacher.ipynb)
ST_IMAGE_SIZE = 224
T_BACKBONE = "wide_resnet50_2"
S_BACKBONE = "resnet18"
ST_OUT_INDICES = (1, 2, 3)
ST_N_STUDENTS = 3
ST_STUDENT_PRETRAINED = False
ST_EPOCHS = 25
ST_LR = 3e-4
ST_WEIGHT_DECAY = 1e-4
ST_BATCH_SIZE = 16
ST_TOPK_PERCENT = 1.0
ST_SEED = 7

ST_DATASET = cfg.DATASET_DIR  # use the BiRefNet-preprocessed dataset (same as main pipeline)
ST_SAVE_DIR = cfg.OUTPUT_DIR / "student_teacher_models"
OUT_RESULTS = cfg.OUTPUT_DIR / "table_results" / "student_teacher.json"


def list_classes() -> list[str]:
    return sorted(
        d.name for d in ST_DATASET.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    )


def checkpoint_path(class_name: str) -> Path:
    return ST_SAVE_DIR / f"student_teacher_{class_name}_T-{T_BACKBONE}_S-{S_BACKBONE}.pt"


def train_all(classes: list[str]) -> None:
    ST_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        ckpt = checkpoint_path(cls)
        if ckpt.exists():
            print(f"[{cls}] checkpoint exists, skipping training: {ckpt}")
            continue
        print(f"\n=== Training Student-Teacher on {cls} ===")
        set_seed(ST_SEED)
        train_student_teacher(
            path=str(ST_DATASET),
            class_name=cls,
            image_size=ST_IMAGE_SIZE,
            backbone=T_BACKBONE,
            student_backbone=S_BACKBONE,
            out_indices=ST_OUT_INDICES,
            n_students=ST_N_STUDENTS,
            student_pretrained=ST_STUDENT_PRETRAINED,
            epochs=ST_EPOCHS,
            batch_size=ST_BATCH_SIZE,
            lr=ST_LR,
            weight_decay=ST_WEIGHT_DECAY,
            seed=ST_SEED,
            device="cuda" if torch.cuda.is_available() else "cpu",
            topk_percent=ST_TOPK_PERCENT,
        )
        # `train_student_teacher` writes to ./student_teacher_models/ by default;
        # move the resulting file into our outputs dir if needed.
        default_dir = Path("./student_teacher_models")
        default_ckpt = default_dir / f"student_teacher_{cls}_T-{T_BACKBONE}_S-{S_BACKBONE}.pt"
        if default_ckpt.exists() and not ckpt.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            default_ckpt.rename(ckpt)
            print(f"  moved checkpoint → {ckpt}")


def load_checkpoint(ckpt_path: Path, device: str) -> StudentTeacherAD:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state = checkpoint["model"]
    model = StudentTeacherAD(
        teacher_backbone=model_state["teacher_backbone"],
        student_backbone=model_state["student_backbone"],
        out_indices=tuple(model_state["out_indices"]),
        n_students=int(model_state["n_students"]),
        device=device,
    ).to(device)
    for student, student_state in zip(model.students, model_state["students"]):
        student.load_state_dict(student_state)
    for projector, projector_state in zip(
        getattr(model, "student_projectors", []),
        model_state.get("student_projectors", []),
    ):
        projector.load_state_dict(projector_state)
    if model_state.get("score_stats") is not None:
        model.set_score_stats(model_state["score_stats"])
    return model


def make_predict_fn_factory(device: str):
    _, eval_tf = make_transforms(ST_IMAGE_SIZE)

    def factory(class_name: str):
        ckpt = checkpoint_path(class_name)
        model = load_checkpoint(ckpt, device)
        model.eval()

        @torch.no_grad()
        def predict_fn(img_path: Path, img_pil: Image.Image) -> np.ndarray:
            x = eval_tf(img_pil).unsqueeze(0).to(device, non_blocking=True)
            amap = compute_student_teacher_maps(
                model=model, images=x, image_size=ST_IMAGE_SIZE
            )[0]
            return amap.detach().cpu().numpy()
        return predict_fn
    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    classes = list_classes()
    print(f"Classes: {classes}")

    if not args.skip_train:
        train_all(classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Evaluating Student-Teacher ===")
    results = evaluate_all_classes(
        predict_fn_factory=make_predict_fn_factory(device),
        dataset_dir=ST_DATASET,
        classes=classes,
    )
    save_results(results, OUT_RESULTS, method_name="Student-Teacher")


if __name__ == "__main__":
    main()
