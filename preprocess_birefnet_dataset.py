from __future__ import annotations

import argparse
import sys
from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parent
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from src.birefnet import BiRefNetBackgroundRemover, preprocess_dataset_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the ADL dataset with BiRefNet background removal."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./dataset/adl-2025-2026-anomaly-detection"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./dataset/adl-2025-2026-anomaly-detection_birefnet"),
        help="Destination root for foreground-preserved images.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ZhengPeng7/BiRefNet",
        help="BiRefNet checkpoint name from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--fill",
        type=str,
        default="black",
        choices=("white", "black", "gray", "mean"),
        help="Background fill color after masking.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask threshold used to keep foreground pixels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for inference, for example cuda or cpu.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=448,
        help="BiRefNet input size used by the preprocessing model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remover = BiRefNetBackgroundRemover(
        model_name=args.model_name,
        input_size=args.input_size,
        device=args.device,
    )
    saved_paths = preprocess_dataset_tree(
        input_root=args.input_root,
        output_root=args.output_root,
        remover=remover,
        fill=args.fill,
        threshold=args.threshold,
    )
    print(f"Saved {len(saved_paths)} images to {args.output_root}")


if __name__ == "__main__":
    main()
