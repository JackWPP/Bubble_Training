"""Debug: process exactly one large image with timing."""
import json, sys, time, random
from pathlib import Path
from PIL import Image
import torch, numpy as np, cv2
from transformers import Sam3Model, Sam3Processor

MODEL_PATH = "/home/xgx/sam3"
MAX_DIM = 1024
MAX_CONCEPT = 20

# Load big_fengchao first image
coco = json.load(open("/home/xgx/Bubble_Training/Dataset/big_fengchao/annotations/instances_default.json"))
img_info = coco["images"][0]
img_path = "/home/xgx/Bubble_Training/Dataset/big_fengchao/images/default/" + img_info["file_name"]
print(f"Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")

anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
print(f"Annotations: {len(anns)}")

image = Image.open(img_path).convert("RGB")
orig_w, orig_h = image.size

# Resize
scale = 1.0
if max(orig_w, orig_h) > MAX_DIM:
    scale = MAX_DIM / max(orig_w, orig_h)
    image = image.resize((int(round(orig_w*scale)), int(round(orig_h*scale))), Image.LANCZOS)
    print(f"Resized: {image.size}, scale={scale:.3f}")

# Load model
t0 = time.time()
model = Sam3Model.from_pretrained(MODEL_PATH).to("cuda").eval()
processor = Sam3Processor.from_pretrained(MODEL_PATH)
print(f"Model loaded: {time.time()-t0:.1f}s")

# Phase 1: Concept with 20 random boxes
t0 = time.time()
sampled = random.sample(anns, min(MAX_CONCEPT, len(anns)))
boxes_xyxy = []
for a in sampled:
    x, y, w, h = a["bbox"]
    boxes_xyxy.append([x*scale, y*scale, (x+w)*scale, (y+h)*scale])

print(f"Concept pass with {len(boxes_xyxy)} boxes...")
inputs = processor(images=image, input_boxes=[boxes_xyxy], input_boxes_labels=[[1]*len(boxes_xyxy)], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(orig_h, orig_w)])[0]
print(f"  Concept pass: {time.time()-t0:.1f}s, masks={len(results['masks'])}")

# Phase 2: Batch fallback for ALL 313 boxes
unmatched_anns = anns[:]  # Pretend all unmatched
BATCH = 20
fallback_time = 0
total_fallback = 0

for start in range(0, len(unmatched_anns), BATCH):
    batch = unmatched_anns[start:start+BATCH]
    batch_boxes = []
    for a in batch:
        x, y, w, h = a["bbox"]
        batch_boxes.append([x*scale, y*scale, (x+w)*scale, (y+h)*scale])

    t1 = time.time()
    try:
        inputs = processor(images=image, input_boxes=[batch_boxes], input_boxes_labels=[[1]*len(batch_boxes)], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results2 = processor.post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(orig_h, orig_w)])[0]
        total_fallback += len(results2["masks"])
    except Exception as e:
        print(f"  Batch {start}: ERROR {e}")
    fallback_time += time.time() - t1

    if start == 0:
        print(f"  Batch 0/{len(unmatched_anns)//BATCH+1}: {time.time()-t1:.1f}s, masks={len(results2.get('masks',[]))}")

print(f"Total fallback: {fallback_time:.1f}s for {len(unmatched_anns)//BATCH+1} batches, {total_fallback} masks found")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
