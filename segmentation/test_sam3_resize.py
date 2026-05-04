"""Quick test: SAM3 with image resizing for a large image."""
import json, sys, time
from pathlib import Path
from PIL import Image
import torch, numpy as np, cv2
from transformers import Sam3Model, Sam3Processor

MODEL = "/mnt/g/Files/SAM3/sam3"
MAX_DIM = 1536

# Load a large image from big_fengchao
coco = json.load(open("/mnt/g/Bubble_Train/Dataset/big_fengchao/annotations/instances_default.json"))
img_info = coco["images"][0]
img_path = "/mnt/g/Bubble_Train/Dataset/big_fengchao/images/default/" + img_info["file_name"]
print(f"Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")

image = Image.open(img_path).convert("RGB")
orig_w, orig_h = image.size

# Get all bboxes
anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
print(f"Annotations: {len(anns)}")

# Resize
scale = 1.0
if max(orig_w, orig_h) > MAX_DIM:
    scale = MAX_DIM / max(orig_w, orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized to {new_w}x{new_h}, scale={scale:.3f}")

boxes_xyxy = []
for a in anns:
    x, y, w, h = a["bbox"]
    boxes_xyxy.append([x * scale, y * scale, (x+w) * scale, (y+h) * scale])

print(f"Using {len(boxes_xyxy)} boxes...")

model = Sam3Model.from_pretrained(MODEL).to("cuda").eval()
processor = Sam3Processor.from_pretrained(MODEL)

t0 = time.time()
inputs = processor(images=image, input_boxes=[boxes_xyxy], input_boxes_labels=[[1]*len(boxes_xyxy)], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(orig_h, orig_w)])[0]
elapsed = time.time() - t0

masks = len(results["masks"])
boxes = len(results.get("boxes", []))
print(f"Time: {elapsed:.1f}s, Masks: {masks}, Boxes: {boxes}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
