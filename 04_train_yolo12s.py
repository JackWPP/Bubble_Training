"""
YOLO12 fine-tuning script for the integrated bubble dataset.
"""

import multiprocessing
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

PRETRAINED_WEIGHTS = os.getenv("BUBBLE_PRETRAINED_WEIGHTS", str(ROOT / "yolo12s.pt"))
DATA_CONFIG = os.getenv("BUBBLE_DATA_CONFIG", str(ROOT / "yolo_dataset_integrated" / "bubble.yaml"))
PROJECT_DIR = os.getenv("BUBBLE_PROJECT_DIR", str(ROOT / "runs"))
DEVICE = os.getenv("BUBBLE_DEVICE", "0")
EXPERIMENT_NAME = "yolo12s_finetune"


if __name__ == "__main__":
    multiprocessing.freeze_support()

    model = YOLO(PRETRAINED_WEIGHTS)

    results = model.train(
        data=DATA_CONFIG,
        epochs=150,
        imgsz=640,
        batch=8,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        warmup_bias_lr=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=30,
        save=True,
        save_period=10,
        device=DEVICE,
        workers=4,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=5,
        amp=True,
        val=True,
        plots=True,
    )
