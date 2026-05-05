"""YOLO-seg training entry point for bubble instance segmentation.

Usage (from project root):
    # Smoke test
    python segmentation/scripts/train_seg.py \
        --model segmentation/configs/models/bubble_yolo11s_seg.yaml \
        --cfg segmentation/configs/train/seg_smoke.yaml

    # Full training
    python segmentation/scripts/train_seg.py \
        --model segmentation/configs/models/bubble_yolo11s_seg.yaml \
        --cfg segmentation/configs/train/seg_full.yaml \
        --name bubble_seg_baseline

    # With P3LCRefine
    python segmentation/scripts/train_seg.py \
        --model segmentation/configs/models/bubble_yolo11s_seg_p3lc.yaml \
        --cfg segmentation/configs/train/seg_full.yaml \
        --name bubble_seg_p3lc
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics_custom import register_bubble_modules


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO-seg for bubble segmentation")
    parser.add_argument("--model", type=str, required=True, help="Model YAML path")
    parser.add_argument("--cfg", type=str, required=True, help="Training config YAML path")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--pretrained", type=str, default="yolo11s-seg.pt", help="Pretrained weights")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--nwd", type=float, default=None, help="NWD loss weight (0.05 from best detection model)")
    parser.add_argument("--nwd-constant", type=float, default=12.8, help="NWD constant C")
    parser.add_argument("--iou-type", type=str, default="CIoU", help="IoU variant for NWD loss")
    parser.add_argument("--dice", type=float, default=None, help="Dice loss weight for mask (e.g. 0.3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register bubble custom modules (P3LCRefine, etc.)
    register_bubble_modules()

    # Enable NWD loss if requested
    if args.nwd is not None and args.nwd > 0:
        from ultralytics_custom.bubble_loss import enable_nwd_loss
        enable_nwd_loss(
            nwd_weight=args.nwd,
            nwd_constant=args.nwd_constant,
            iou_type=args.iou_type,
        )
        print(f"NWD loss enabled: weight={args.nwd}, C={args.nwd_constant}, IoU={args.iou_type}")

    # Enable Dice+BCE hybrid mask loss if requested
    if args.dice is not None and args.dice > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dice_loss import enable_dice_loss
        enable_dice_loss(dice_weight=args.dice)

    # Load training config overrides
    overrides = {
        "model": args.model,
        "data": "/home/xgx/Bubble_Training/segmentation/datasets/paper_v4_seg/bubble_seg.yaml",
        "device": args.device,
    }

    if args.name:
        overrides["name"] = args.name
        overrides["project"] = "segmentation/runs/bubble_seg"

    # Determine if pretrained model exists
    pretrained_path = args.pretrained
    if pretrained_path and not Path(pretrained_path).exists():
        print(f"Pretrained model {pretrained_path} not found locally, Ultralytics will download it.")
    elif pretrained_path and Path(pretrained_path).exists():
        overrides["pretrained"] = pretrained_path

    print(f"Model: {args.model}")
    print(f"Config: {args.cfg}")
    print(f"Pretrained: {pretrained_path}")

    # Train
    model = YOLO(args.model)
    model.train(cfg=args.cfg, **overrides)


if __name__ == "__main__":
    main()
