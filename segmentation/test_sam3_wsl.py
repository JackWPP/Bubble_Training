"""SAM3 verification script — run in WSL: python3 test_sam3_wsl.py"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# ====== CONFIGURE THESE PATHS FOR YOUR SERVER ======
MODEL_PATH = "/mnt/g/Files/SAM3/sam3"
DATASET_DIR = Path("/mnt/g/Bubble_Train/Dataset")
# ===================================================
TEST_SOURCE = "20+40"

def load_first_image_and_boxes(coco_path: Path):
    with open(coco_path) as f:
        coco = json.load(f)
    img_info = coco["images"][0]
    img_path = coco_path.parent.parent / "images" / "default" / img_info["file_name"]
    image = Image.open(str(img_path)).convert("RGB")
    anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
    boxes_xyxy = []
    for a in anns:
        x, y, w, h = a["bbox"]
        boxes_xyxy.append([x, y, x + w, y + h])
    return image, boxes_xyxy, img_info

def main():
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {mem:.1f} GB")

    print(f"\nLoading SAM3 from {MODEL_PATH} ...")
    # Don't use float16 — let the model handle dtype internally
    model = Sam3Model.from_pretrained(MODEL_PATH).to("cuda").eval()
    processor = Sam3Processor.from_pretrained(MODEL_PATH)
    print("SAM3 loaded successfully.")

    coco_path = DATASET_DIR / TEST_SOURCE / "annotations" / "instances_default.json"
    if not coco_path.exists():
        print(f"ERROR: {coco_path} not found"); sys.exit(1)

    image, boxes_xyxy, img_info = load_first_image_and_boxes(coco_path)
    print(f"\nImage: {img_info['file_name']} ({img_info['width']}x{img_info['height']}), boxes: {len(boxes_xyxy)}")

    test_boxes = boxes_xyxy[:3]
    print(f"Testing {len(test_boxes)} boxes: {test_boxes}")

    inputs = processor(
        images=image,
        input_boxes=[test_boxes],
        input_boxes_labels=[[1] * len(test_boxes)],
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=[image.size[::-1]]
    )[0]

    masks = results["masks"]  # list of tensors
    scores = results.get("scores", [torch.tensor(0.0)] * len(masks))
    print(f"\nMasks: {len(masks)}")

    # Convert masks to numpy
    masks_np = [m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in masks]

    # Visualize
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (mask, score_t) in enumerate(zip(masks_np, scores)):
        color = colors[i % len(colors)]
        binary = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, color, 2)
        if contours:
            cnt = contours[0]
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            score = score_t.item() if isinstance(score_t, torch.Tensor) else score_t
            print(f"  Mask {i}: score={score:.3f}, pts={len(cnt)}, simplified={len(approx)}")

    out = Path("/mnt/g/Bubble_Train/segmentation/test_sam3_output.jpg")
    cv2.imwrite(str(out), img_bgr)
    print(f"\nSaved: {out}")
    print("PASSED.")

if __name__ == "__main__":
    main()
