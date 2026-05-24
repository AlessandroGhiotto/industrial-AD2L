"""Read per-method JSON files from outputs/table_results/ and emit a LaTeX
snippet ready to paste into the Table 1 environment in final/report/main.tex.

Expected files (any missing row is filled with `--`):
    autoencoder.json
    student_teacher.json
    patchcore.json
    unet_no_birefnet.json
    unet.json
    ensemble.json

Each JSON has shape:
    {"method": "...",
     "per_class": {"class_01": 0.42, ..., "class_08": 0.55, "mean": 0.48}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = REPO_ROOT / "final"
sys.path.insert(0, str(FINAL_DIR))
from src import config_final as cfg  # noqa: E402

RESULTS_DIR = cfg.OUTPUT_DIR / "table_results"

# Order of rows in the LaTeX table, matching the report draft.
ROWS = [
    ("autoencoder.json",        "AE"),
    ("student_teacher.json",    "ST"),
    ("patchcore.json",          "PC (DINOv2-base)"),
    ("unet_no_birefnet.json",   r"U-Net$^{-}$"),
    ("unet.json",               "U-Net"),
    ("ensemble.json",           r"\textbf{Ens.}"),
]
CLASSES = [f"class_0{i}" for i in range(1, 9)]


def fmt(v) -> str:
    try:
        return f"{float(v) * 100:.1f}"
    except (TypeError, ValueError):
        return "--"


def load(fname: str) -> dict[str, float] | None:
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())["per_class"]


def build_rows() -> list[str]:
    out = []
    for fname, label in ROWS:
        data = load(fname)
        if data is None:
            cells = ["--"] * (len(CLASSES) + 1)
        else:
            cells = [fmt(data.get(c)) for c in CLASSES]
            cells.append(fmt(data.get("mean")))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


def main() -> None:
    rows = build_rows()
    header = " & ".join(["Method"] + [f"C{i}" for i in range(1, 9)] + ["Mean"])
    print("% --- paste this body into the tabular env of Table 1 ---")
    print(header + r" \\")
    print(r"\midrule")
    for r in rows[:-1]:
        print(r)
    print(r"\midrule")
    print(rows[-1])
    print("% -------------------------------------------------------")


if __name__ == "__main__":
    main()
