"""Generate segmentation masks via SAM3 box prompts.

Usage:
    cd /home/xgx/Bubble_Training/segmentation
    python generate_masks.py

Input:  Dataset/*/annotations/instances_default.json (bbox only)
Output: Dataset/*/annotations/instances_default_segmented.json (bbox + segmentation)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Sam3Model, Sam3Processor

# ====== CONFIGURE THESE PATHS FOR YOUR SERVER ======
MODEL_PATH = "/home/xgx/sam3"       # Directory containing model.safetensors + config.json
DATASET_DIR = Path("/home/xgx/Bubble_Training/Dataset")  # Directory containing COCO JSON data sources
# ===================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sources to process
SOURCES = [
    "20+40",
    "60+80",
    "big_fengchao",
    "bubble_1",
    "bubble_fc",
    "bubble_pad",
    "job_13_dataset_2026_04_30_19_34_23_coco 1.0",
]

IOU_THRESHOLD = 0.3       # Min IoU to accept a SAM3 mask as matching a bbox
POLYGON_EPSILON = 1.0     # approxPolyDP epsilon (pixels in original image)
SIMPLIFY_MAX_PTS = 50     # Max polygon vertices after simplification
MIN_BUBBLE_AREA_PX = 4    # Minimum mask area in pixels
CONF_THRESHOLD = 0.3      # Minimum SAM3 confidence score
MAX_IMAGE_DIM = 1024      # Resize images whose longest side exceeds this (saves VRAM)
MAX_CONCEPT_BOXES = 20    # Max boxes to use as concept exemplars (limits attention complexity)


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] → (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def box_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_to_polygon(mask: np.ndarray) -> list[float] | None:
    """Convert binary mask to COCO polygon (flattened list). Returns None if no valid polygon."""
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take largest contour
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_BUBBLE_AREA_PX:
        return None
    # Simplify
    perimeter = cv2.arcLength(cnt, True)
    epsilon = POLYGON_EPSILON * (perimeter / 100.0) if perimeter > 0 else POLYGON_EPSILON
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Limit vertex count
    if len(approx) > SIMPLIFY_MAX_PTS:
        # Increase epsilon until under limit
        while len(approx) > SIMPLIFY_MAX_PTS:
            epsilon *= 1.3
            approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Flatten to COCO polygon format: [x1, y1, x2, y2, ...]
    flat = [float(v) for pt in approx for v in pt[0]]
    if len(flat) < 6:
        return None
    return flat


def match_masks_to_bboxes(
    sam3_boxes: list[list[float]],
    sam3_scores: list[float],
    annotation_bboxes: list[list[float]],
) -> list[int | None]:
    """Match each SAM3 mask to an annotation bbox by IoU.

    Returns: matched_sam3_idx_for_each_annotation (None = no match)
    """
    matched = [None] * len(annotation_bboxes)
    used = set()

    # Build IoU matrix
    for sam3_idx, sam3_box in enumerate(sam3_boxes):
        if sam3_scores[sam3_idx] < CONF_THRESHOLD:
            continue
        best_ann_idx = -1
        best_iou = 0.0
        for ann_idx, ann_bbox in enumerate(annotation_bboxes):
            ann_box = coco_bbox_to_xyxy(ann_bbox)
            iou = box_iou(tuple(sam3_box), ann_box)
            if iou > best_iou:
                best_iou = iou
                best_ann_idx = ann_idx
        if best_iou >= IOU_THRESHOLD and best_ann_idx >= 0:
            if matched[best_ann_idx] is None or sam3_scores[sam3_idx] > sam3_scores[matched[best_ann_idx]]:
                matched[best_ann_idx] = sam3_idx

    return matched


def process_source(source_name: str, model: Sam3Model, processor: Sam3Processor) -> dict:
    """Process one COCO data source: add segmentation polygons to all annotations."""
    coco_path = DATASET_DIR / source_name / "annotations" / "instances_default.json"
    if not coco_path.exists():
        print(f"  SKIP: {coco_path} not found")
        return {"status": "skipped", "reason": "not found"}

    out_path = DATASET_DIR / source_name / "annotations" / "instances_default_segmented.json"

    # Resume: skip only if output exists AND all annotations have segmentation
    if out_path.exists():
        with open(out_path) as f:
            coco = json.load(f)
        existing_segs = sum(1 for a in coco.get("annotations", []) if a.get("segmentation") and a["segmentation"] != [[]])
        total = len(coco.get("annotations", []))
        if existing_segs >= total:
            print(f"  SKIP: already processed ({existing_segs}/{total} have segmentation)")
            return {"status": "skipped", "reason": "already done", "existing_segmented": existing_segs}
        print(f"  Resume: {existing_segs}/{total} already have segmentation, processing remaining")
    else:
        with open(coco_path) as f:
            coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    print(f"  {len(images)} images, {len(annotations)} annotations")

    total_annotations = len(annotations)
    matched_count = 0
    fallback_count = 0
    failed_count = 0

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_info in tqdm(images, desc=f"  {source_name}"):
        img_id = img_info["id"]
        img_anns = anns_by_image.get(img_id, [])
        if not img_anns:
            continue

        img_path = DATASET_DIR / source_name / "images" / "default" / img_info["file_name"]
        if not img_path.exists():
            continue

        image = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = image.size

        # Resize large images to save VRAM (scale factor applied to bboxes)
        scale_factor = 1.0
        if max(orig_w, orig_h) > MAX_IMAGE_DIM:
            scale_factor = MAX_IMAGE_DIM / max(orig_w, orig_h)
            new_w = int(round(orig_w * scale_factor))
            new_h = int(round(orig_h * scale_factor))
            image = image.resize((new_w, new_h), Image.LANCZOS)

        annotation_bboxes = [ann["bbox"] for ann in img_anns]

        # Phase 1: Concept segmentation — use a SUBSET of bboxes as exemplars
        # (Too many prompts causes O(n²) attention blowup in SAM3)
        import random as _random
        sampled_bboxes = annotation_bboxes[:]
        if len(sampled_bboxes) > MAX_CONCEPT_BOXES:
            sampled_bboxes = _random.sample(sampled_bboxes, MAX_CONCEPT_BOXES)
        xyxy_boxes = []
        for b in sampled_bboxes:
            x, y, w, h = b
            xyxy_boxes.append([
                x * scale_factor,
                y * scale_factor,
                (x + w) * scale_factor,
                (y + h) * scale_factor,
            ])
        try:
            inputs = processor(
                images=image,
                input_boxes=[xyxy_boxes],
                input_boxes_labels=[[1] * len(xyxy_boxes)],
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs, threshold=CONF_THRESHOLD, mask_threshold=0.5,
                target_sizes=[(orig_h, orig_w)]
            )[0]

            sam3_masks = [m.cpu().numpy() for m in results["masks"]]
            sam3_boxes = results.get("boxes", [])
            sam3_scores = [s.item() if isinstance(s, torch.Tensor) else s for s in results.get("scores", [0.0] * len(sam3_masks))]

        except Exception as e:
            print(f"\n  ERROR: {img_info['file_name']}: {e}")
            sam3_masks, sam3_boxes, sam3_scores = [], [], []

        # Free GPU tensors from concept phase
        for _v in ("outputs", "results", "inputs"):
            _obj = locals().get(_v)
            if _obj is not None:
                del _obj
        torch.cuda.empty_cache()

        # Phase 2: Match SAM3 masks to annotation bboxes
        matched_indices = match_masks_to_bboxes(sam3_boxes, sam3_scores, annotation_bboxes)

        # Phase 3: Assign matched polygons
        for ann_idx, (ann, sam3_idx) in enumerate(zip(img_anns, matched_indices)):
            if sam3_idx is not None:
                polygon = mask_to_polygon(sam3_masks[sam3_idx])
                if polygon:
                    ann["segmentation"] = [polygon]
                    matched_count += 1

        # Phase 4: Batch fallback for unmatched boxes (group into batches of MAX_CONCEPT_BOXES)
        unmatched = [(i, ann) for i, (ann, mi) in enumerate(zip(img_anns, matched_indices))
                     if mi is None and not ann.get("segmentation")]
        if unmatched:
            for batch_start in range(0, len(unmatched), MAX_CONCEPT_BOXES):
                batch = unmatched[batch_start:batch_start + MAX_CONCEPT_BOXES]
                batch_boxes_scaled = []
                batch_anns = []
                for _, ann in batch:
                    x, y, w, h = ann["bbox"]
                    batch_boxes_scaled.append([
                        x * scale_factor, y * scale_factor,
                        (x + w) * scale_factor, (y + h) * scale_factor,
                    ])
                    batch_anns.append(ann)

                try:
                    inputs = processor(
                        images=image,
                        input_boxes=[batch_boxes_scaled],
                        input_boxes_labels=[[1] * len(batch_boxes_scaled)],
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    results = processor.post_process_instance_segmentation(
                        outputs, threshold=CONF_THRESHOLD, mask_threshold=0.5,
                        target_sizes=[(orig_h, orig_w)]
                    )[0]
                    b_masks = [m.cpu().numpy() for m in results["masks"]]
                    b_boxes = results.get("boxes", [])
                    b_scores = [s.item() if isinstance(s, torch.Tensor) else s for s in results.get("scores", [0.0] * len(b_masks))]

                    # Match each annotation in batch to best mask
                    for ann, orig_bbox in zip(batch_anns, [ann["bbox"] for _, ann in batch]):
                        target = coco_bbox_to_xyxy(orig_bbox)
                        best_bidx = -1
                        best_biou = 0.0
                        for j, pb in enumerate(b_boxes):
                            iou = box_iou(tuple(pb), target)
                            if iou > best_biou:
                                best_biou = iou
                                best_bidx = j
                        if best_bidx >= 0 and best_biou >= IOU_THRESHOLD:
                            poly = mask_to_polygon(b_masks[best_bidx])
                            if poly:
                                ann["segmentation"] = [poly]
                                fallback_count += 1
                except Exception as e:
                    pass  # Individual single-box fallback below
                finally:
                    # Free tensors after each batch
                    for v in ["outputs", "results", "inputs", "b_masks", "b_boxes", "b_scores"]:
                        _v = locals().pop(v, None)
                        if _v is not None:
                            del _v
                    torch.cuda.empty_cache()

        # Phase 5: Last-resort single-box fallback for any remaining unmatched
        for ann in img_anns:
            if ann.get("segmentation"):
                continue
            x, y, w, h = ann["bbox"]
            xyxy_box = [
                x * scale_factor, y * scale_factor,
                (x + w) * scale_factor, (y + h) * scale_factor,
            ]
            poly = single_box_inference(image, xyxy_box, ann, model, processor, orig_w, orig_h, scale_factor)
            if poly:
                ann["segmentation"] = [poly]
                fallback_count += 1
            else:
                failed_count += 1

        # Free image and mask tensors, run garbage collection
        del image, sam3_masks, sam3_boxes, sam3_scores
        gc.collect()
        torch.cuda.empty_cache()

        # Incremental save after each image (survives OOM kills)
        out_path = DATASET_DIR / source_name / "annotations" / "instances_default_segmented.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

    # Final save
    out_path = DATASET_DIR / source_name / "annotations" / "instances_default_segmented.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    print(f"  Results: {matched_count} matched, {fallback_count} fallback, {failed_count} failed (of {total_annotations})")
    return {"status": "ok", "matched": matched_count, "fallback": fallback_count, "failed": failed_count}


def single_box_inference(
    image: Image.Image,
    box_xyxy: list[float],
    ann: dict,
    model: Sam3Model,
    processor: Sam3Processor,
    orig_w: int,
    orig_h: int,
    scale_factor: float = 1.0,
) -> list[float] | None:
    """Run SAM3 with a single box prompt, return polygon for best-matching mask.
    box_xyxy is in resized image coordinates; output polygon is in original coordinates."""
    try:
        inputs = processor(
            images=image,
            input_boxes=[[box_xyxy]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=CONF_THRESHOLD, mask_threshold=0.5,
            target_sizes=[(orig_h, orig_w)]
        )[0]

        masks = [m.cpu().numpy() for m in results["masks"]]
        boxes = results.get("boxes", [])
        scores = [s.item() if isinstance(s, torch.Tensor) else s for s in results.get("scores", [0.0] * len(masks))]

        if not masks:
            return None

        # Pick best mask by IoU to input box (scale target to original coords)
        best_idx = 0
        best_iou = 0.0
        target_box = tuple(v / scale_factor for v in box_xyxy) if scale_factor != 1.0 else tuple(box_xyxy)
        for i, pred_box in enumerate(boxes):
            iou = box_iou(tuple(pred_box), target_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD / 2:  # Lower threshold for fallback
            return mask_to_polygon(masks[best_idx])
        # As last resort, take highest-scoring mask
        if masks:
            best_by_score = max(range(len(scores)), key=lambda i: scores[i])
            return mask_to_polygon(masks[best_by_score])
        return None
    except Exception as e:
        print(f"\n  ERROR single-box {ann.get('id', '?')}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 mask generation for bubble datasets")
    parser.add_argument("--source", type=str, default=None, help="Process single source only (e.g. '20+40')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    args = parser.parse_args()

    sources = [args.source] if args.source else SOURCES
    if args.dry_run:
        for s in sources:
            coco_path = DATASET_DIR / s / "annotations" / "instances_default.json"
            if coco_path.exists():
                with open(coco_path) as f:
                    coco = json.load(f)
                print(f"{s}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
            else:
                print(f"{s}: NOT FOUND")
        return

    print(f"Device: {DEVICE}")
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if DEVICE == "cuda":
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {mem:.1f} GB")

    print(f"\nLoading SAM3 from {MODEL_PATH} ...")
    model = Sam3Model.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True).to(DEVICE).eval()
    processor = Sam3Processor.from_pretrained(MODEL_PATH)
    print("SAM3 loaded.\n")

    summary = {}
    total_start = time.time()

    for source in sources:
        print(f"\n{'='*60}")
        print(f"Processing: {source}")
        print(f"{'='*60}")
        src_start = time.time()
        result = process_source(source, model, processor)
        elapsed = time.time() - src_start
        result["elapsed_s"] = round(elapsed, 1)
        summary[source] = result
        print(f"  Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE. Total time: {total_elapsed/60:.1f} min")
    print(f"Summary: {json.dumps(summary, indent=2)}")

    # Save summary
    summary_path = Path("/home/xgx/Bubble_Training/segmentation/generate_masks_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
