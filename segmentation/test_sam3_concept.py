"""Quick test: SAM3 concept segmentation mode with ALL boxes as prompts for one image."""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_PATH = "/home/xgx/sam3"
DATASET_DIR = Path("/home/xgx/Bubble_Training/Dataset")
SOURCE = "20+40"

# Load COCO
coco_path = DATASET_DIR / SOURCE / "annotations" / "instances_default.json"
with open(coco_path) as f:
    coco = json.load(f)

img_info = coco["images"][0]
img_path = DATASET_DIR / SOURCE / "images" / "default" / img_info["file_name"]
image = Image.open(str(img_path)).convert("RGB")
orig_w, orig_h = image.size
print(f"Image: {img_info['file_name']} ({orig_w}x{orig_h})")

anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
print(f"Annotations: {len(anns)}")

# Convert all bboxes to xyxy
boxes_xyxy = []
for a in anns:
    x, y, w, h = a["bbox"]
    boxes_xyxy.append([x, y, x + w, y + h])

print(f"Using {len(boxes_xyxy)} bbox prompts...")

model = Sam3Model.from_pretrained(MODEL_PATH).to("cuda").eval()
processor = Sam3Processor.from_pretrained(MODEL_PATH)

inputs = processor(
    images=image,
    input_boxes=[boxes_xyxy],
    input_boxes_labels=[[1] * len(boxes_xyxy)],
    return_tensors="pt",
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs, threshold=0.3, mask_threshold=0.5,
    target_sizes=[(orig_h, orig_w)]
)[0]

masks = [m.cpu().numpy() for m in results["masks"]]
pred_boxes = results.get("boxes", [])
scores = [s.item() for s in results.get("scores", [0]*len(masks))]

print(f"\nSAM3 masks: {len(masks)}")
print(f"Annotation boxes: {len(anns)}")

# IoU matching
def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

matched = 0
unmatched = 0
for ann in anns:
    x, y, w, h = ann["bbox"]
    ann_box = (x, y, x+w, y+h)
    best_iou = 0
    best_idx = -1
    for j, pb in enumerate(pred_boxes):
        iou = box_iou(tuple(pb), ann_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = j
    if best_iou >= 0.3:
        matched += 1
    else:
        unmatched += 1
        print(f"  Unmatched: bbox={ann_box}, best_iou={best_iou:.3f}, best_idx={best_idx}")

print(f"\nMatched: {matched}/{len(anns)}, Unmatched: {unmatched}/{len(anns)}")

# Visualize matching
img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for ann in anns:
    x, y, w, h = ann["bbox"]
    cv2.rectangle(img_bgr, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 1)

for i, (mask, pred_box) in enumerate(zip(masks, pred_boxes)):
    binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours, -1, (255, 0, 0), 1)

out = Path("/home/xgx/Bubble_Training/segmentation/test_concept_match.jpg")
cv2.imwrite(str(out), img_bgr)
print(f"Saved: {out}")
