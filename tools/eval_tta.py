"""Evaluate selected weights with optional Ultralytics test-time augmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules


def parse_device(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def metric_dict(metrics: Any) -> dict[str, Any]:
    data = getattr(metrics, "results_dict", {}) or {}
    return {str(key): float(value) for key, value in data.items()}


def run_eval(
    weight: Path,
    data: Path,
    split: str,
    imgsz: int,
    device: str | int,
    project: Path,
    name: str,
    augment: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weight))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        device=device,
        augment=augment,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=False,
    )
    return metric_dict(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "yolo_dataset_paper_v4" / "bubble.yaml")
    parser.add_argument("--ood-data", type=Path, default=ROOT / "yolo_dataset_grouped" / "bubble.yaml")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bubble_paper_v4_tta_eval")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    register_bubble_modules()

    weight = args.weight if args.weight.is_absolute() else ROOT / args.weight
    data = args.data if args.data.is_absolute() else ROOT / args.data
    ood_data = args.ood_data if args.ood_data.is_absolute() else ROOT / args.ood_data
    project = args.project if args.project.is_absolute() else ROOT / args.project
    device = parse_device(args.device)

    results: dict[str, Any] = {
        "name": args.name,
        "weight": str(weight),
        "imgsz": args.imgsz,
        "main_test": {},
        "ood_test": {},
    }
    for augment in (False, True):
        suffix = "tta" if augment else "plain"
        results["main_test"][suffix] = run_eval(
            weight, data, "test", args.imgsz, device, project, f"{args.name}_main_{suffix}", augment
        )
        results["ood_test"][suffix] = run_eval(
            weight, ood_data, "test", args.imgsz, device, project, f"{args.name}_ood_{suffix}", augment
        )

    out_path = project / f"{args.name}.json"
    project.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[summary] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
