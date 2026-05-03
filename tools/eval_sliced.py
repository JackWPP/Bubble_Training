"""Evaluate YOLO weights with full-image and sliced inference.

This tool is intentionally dependency-light. It uses Ultralytics for model
inference, then computes AP from YOLO-format labels so sliced inference can be
compared with ordinary full-image inference on the same images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics_custom import register_bubble_modules


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path


def parse_device(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    raw = Path(str(data.get("path", data_yaml.parent)))
    if raw.is_absolute():
        return raw
    candidates = [ROOT / raw, data_yaml.parent / raw, data_yaml.parent]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / raw).resolve()


def resolve_split_dir(data_yaml: Path, split: str) -> Path:
    data = load_yaml(data_yaml)
    root = resolve_dataset_root(data_yaml, data)
    split_value = data.get(split)
    if split_value is None:
        raise KeyError(f"{data_yaml} has no split {split!r}")
    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path
    return (root / split_path).resolve()


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def collect_records(data_yaml: Path, split: str, limit: int | None) -> list[ImageRecord]:
    image_dir = resolve_split_dir(data_yaml, split)
    if not image_dir.exists():
        raise FileNotFoundError(f"Split image directory does not exist: {image_dir}")
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        images = images[:limit]
    return [ImageRecord(path, label_for_image(path)) for path in images]


def load_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def tile_origins(width: int, height: int, slice_size: int, overlap: int) -> list[tuple[int, int]]:
    if slice_size <= 0:
        raise ValueError("--slice-size must be positive")
    if overlap < 0 or overlap >= slice_size:
        raise ValueError("--overlap must be >= 0 and smaller than --slice-size")
    stride = slice_size - overlap

    def axis_positions(length: int) -> list[int]:
        if length <= slice_size:
            return [0]
        positions = list(range(0, max(length - slice_size, 0) + 1, stride))
        edge = length - slice_size
        if positions[-1] != edge:
            positions.append(edge)
        return positions

    return [(x, y) for y in axis_positions(height) for x in axis_positions(width)]


def xyxy_iou(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter_x1 = torch.maximum(box[0], boxes[:, 0])
    inter_y1 = torch.maximum(box[1], boxes[:, 1])
    inter_x2 = torch.minimum(box[2], boxes[:, 2])
    inter_y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def nms_class_agnostic(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    order = scores_t.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0 and len(keep) < max_det:
        idx = int(order[0])
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = xyxy_iou(boxes_t[idx], boxes_t[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def clip_boxes_with_attrs(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return (
            boxes.reshape(0, 4).astype(np.float32),
            scores.reshape(0).astype(np.float32),
            classes.reshape(0).astype(np.int64),
        )
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    keep = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
    return boxes[keep], scores[keep], classes[keep]


def predict_batch(model: Any, images: list[np.ndarray], imgsz: int, conf: float, iou: float, max_det: int, device: str | int):
    return model.predict(
        source=images,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
        save=False,
        stream=False,
    )


def boxes_from_result(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    boxes = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
    classes = boxes_obj.cls.detach().cpu().numpy().astype(np.int64)
    return boxes, scores, classes


def full_predict(
    model: Any,
    image: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = predict_batch(model, [image], imgsz, conf, iou, max_det, device)[0]
    boxes, scores, classes = boxes_from_result(result)
    return clip_boxes_with_attrs(boxes, scores, classes, image.shape[1], image.shape[0])


def sliced_predict(
    model: Any,
    image: np.ndarray,
    imgsz: int,
    conf: float,
    pred_iou: float,
    merge_iou: float,
    max_det: int,
    device: str | int,
    slice_size: int,
    overlap: int,
    include_full: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    if include_full:
        crops.append(image)
        offsets.append((0, 0))
    for x, y in tile_origins(width, height, slice_size, overlap):
        crops.append(image[y : min(y + slice_size, height), x : min(x + slice_size, width)])
        offsets.append((x, y))

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    for result, (x, y) in zip(predict_batch(model, crops, imgsz, conf, pred_iou, max_det, device), offsets, strict=True):
        boxes, scores, classes = boxes_from_result(result)
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] += x
        boxes[:, [1, 3]] += y
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    boxes, scores, classes = clip_boxes_with_attrs(boxes, scores, classes, width, height)
    keep = nms_class_agnostic(boxes, scores, merge_iou, max_det)
    return boxes[keep], scores[keep], classes[keep]


def true_positive_matrix(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    tp = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp

    pred_t = torch.as_tensor(pred_boxes, dtype=torch.float32)
    gt_t = torch.as_tensor(gt_boxes, dtype=torch.float32)
    from ultralytics.utils.metrics import box_iou

    ious = box_iou(pred_t, gt_t).cpu().numpy()
    order = np.argsort(-pred_scores)
    for t_idx, threshold in enumerate(IOU_THRESHOLDS):
        matched: set[int] = set()
        for pred_idx in order:
            same_class = gt_classes == pred_classes[pred_idx]
            if not np.any(same_class):
                continue
            candidate_ious = ious[pred_idx].copy()
            candidate_ious[~same_class] = -1.0
            for gt_idx in matched:
                candidate_ious[gt_idx] = -1.0
            best_gt = int(candidate_ious.argmax())
            if candidate_ious[best_gt] >= threshold:
                tp[pred_idx, t_idx] = True
                matched.add(best_gt)
    return tp


def summarize_ap(tp_parts: list[np.ndarray], conf_parts: list[np.ndarray], pred_cls_parts: list[np.ndarray], target_cls: list[int]) -> dict[str, Any]:
    from ultralytics.utils.metrics import ap_per_class

    num_predictions = int(sum(len(part) for part in conf_parts))
    num_labels = int(len(target_cls))
    if num_predictions == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "num_labels": num_labels,
            "num_predictions": 0,
        }

    tp = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    conf = np.concatenate(conf_parts, axis=0) if conf_parts else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(pred_cls_parts, axis=0) if pred_cls_parts else np.zeros((0,), dtype=np.int64)
    target = np.asarray(target_cls, dtype=np.int64)

    _tp_count, _fp_count, p, r, _f1, ap, unique_classes, *_ = ap_per_class(
        tp,
        conf,
        pred_cls,
        target,
        plot=False,
        names={0: "bubble"},
    )
    if len(unique_classes) == 0:
        map50 = map5095 = precision = recall = 0.0
    else:
        precision = float(np.mean(p))
        recall = float(np.mean(r))
        map50 = float(ap[:, 0].mean())
        map5095 = float(ap.mean())
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "num_labels": num_labels,
        "num_predictions": num_predictions,
    }


def evaluate_dataset(
    model: Any,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str | int,
    conf: float,
    pred_iou: float,
    merge_iou: float,
    max_det: int,
    slice_size: int,
    overlap: int,
    limit: int | None,
) -> dict[str, Any]:
    records = collect_records(data_yaml, split, limit)
    plain_tp: list[np.ndarray] = []
    plain_conf: list[np.ndarray] = []
    plain_cls: list[np.ndarray] = []
    sliced_tp: list[np.ndarray] = []
    sliced_conf: list[np.ndarray] = []
    sliced_cls: list[np.ndarray] = []
    target_cls: list[int] = []

    for idx, record in enumerate(records, start=1):
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record.image_path}")
        height, width = image.shape[:2]
        gt_boxes, gt_classes = load_labels(record.label_path, width, height)
        target_cls.extend(int(cls) for cls in gt_classes)

        p_boxes, p_scores, p_classes = full_predict(model, image, imgsz, conf, pred_iou, max_det, device)
        s_boxes, s_scores, s_classes = sliced_predict(
            model,
            image,
            imgsz,
            conf,
            pred_iou,
            merge_iou,
            max_det,
            device,
            slice_size,
            overlap,
            include_full=True,
        )

        plain_tp.append(true_positive_matrix(p_boxes, p_scores, p_classes, gt_boxes, gt_classes))
        plain_conf.append(p_scores)
        plain_cls.append(p_classes)
        sliced_tp.append(true_positive_matrix(s_boxes, s_scores, s_classes, gt_boxes, gt_classes))
        sliced_conf.append(s_scores)
        sliced_cls.append(s_classes)

        if idx == 1 or idx % 50 == 0 or idx == len(records):
            print(
                f"[{data_yaml.name}:{split}] {idx}/{len(records)} "
                f"plain_pred={sum(len(x) for x in plain_conf)} sliced_pred={sum(len(x) for x in sliced_conf)}",
                flush=True,
            )

    plain = summarize_ap(plain_tp, plain_conf, plain_cls, target_cls)
    sliced = summarize_ap(sliced_tp, sliced_conf, sliced_cls, target_cls)
    plain["num_images"] = len(records)
    sliced["num_images"] = len(records)
    return {"plain": plain, "sliced": sliced}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "yolo_dataset_paper_v4" / "bubble.yaml")
    parser.add_argument("--ood-data", type=Path, default=ROOT / "yolo_dataset_grouped" / "bubble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "bubble_paper_v4_sliced_eval")
    parser.add_argument("--name", required=True)
    parser.add_argument("--slice-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.55, help="Per-crop Ultralytics NMS IoU")
    parser.add_argument("--merge-iou", type=float, default=0.55, help="Final class-agnostic merge NMS IoU")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-ood", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    register_bubble_modules()

    from ultralytics import YOLO

    weight = resolve_path(args.weight)
    data = resolve_path(args.data)
    ood_data = resolve_path(args.ood_data)
    project = resolve_path(args.project)
    device = parse_device(args.device)

    model = YOLO(str(weight))
    results: dict[str, Any] = {
        "name": args.name,
        "weight": str(weight),
        "data": str(data),
        "ood_data": str(ood_data),
        "split": args.split,
        "imgsz": args.imgsz,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "conf": args.conf,
        "iou": args.iou,
        "merge_iou": args.merge_iou,
        "max_det": args.max_det,
        "limit": args.limit,
        "main_test": evaluate_dataset(
            model,
            data,
            args.split,
            args.imgsz,
            device,
            args.conf,
            args.iou,
            args.merge_iou,
            args.max_det,
            args.slice_size,
            args.overlap,
            args.limit,
        ),
    }
    if not args.skip_ood:
        results["ood_test"] = evaluate_dataset(
            model,
            ood_data,
            args.split,
            args.imgsz,
            device,
            args.conf,
            args.iou,
            args.merge_iou,
            args.max_det,
            args.slice_size,
            args.overlap,
            args.limit,
        )

    project.mkdir(parents=True, exist_ok=True)
    out_path = project / f"{args.name}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[summary] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
