"""
训练后推理验证脚本
用训练好的最佳权重对新数据做预测并可视化
"""

import os
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = os.getenv("BUBBLE_MODEL_PATH", str(ROOT / "runs" / "yolo12n_finetune" / "weights" / "best.pt"))
SOURCE_DIR = os.getenv("BUBBLE_SOURCE_DIR", str(ROOT / "yolo_dataset_integrated" / "val" / "images"))
OUTPUT_DIR = os.getenv("BUBBLE_OUTPUT_DIR", str(ROOT / "runs" / "predict_val"))

if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
if len(sys.argv) > 2:
    SOURCE_DIR = sys.argv[2]
if len(sys.argv) > 3:
    OUTPUT_DIR = sys.argv[3]

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE_DIR,
    conf=0.25,
    iou=0.45,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(Path(OUTPUT_DIR).parent),
    name=Path(OUTPUT_DIR).name,
    exist_ok=True,
)

for r in results:
    n_boxes = len(r.boxes) if r.boxes is not None else 0
    print(f"  {Path(r.path).name}: {n_boxes} bubbles detected")
