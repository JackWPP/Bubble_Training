# 气泡实例分割 — 实验总结

> 日期: 2026-05-04
> 服务器: 2x Tesla V100 16GB, conda env CNN_2
> 数据: 7 个 COCO 源, 170 张原图, 9,524 个标注框 → SAM3 生成 9,504 个分割标注 (99.79%)

## 1. 流水线概览

```
COCO bbox-only → SAM3 推理 → COCO bbox+polygon → build_dataset.py → YOLO-seg 数据集 → 训练
```

### 1.1 SAM3 分割标注生成
- 脚本: `segmentation/generate_masks.py`
- SAM3 权重: `/home/xgx/sam3/` (model.safetensors 3.4GB)
- 输出: `Dataset/<source>/annotations/instances_default_segmented.json`

### 1.2 数据集构建
- 脚本: `segmentation/build_dataset.py`
- 策略: 70/15/15 分层分割, 640×640 瓦片, 水平翻转增强
- 输出: `segmentation/datasets/paper_v4_seg/` (712 train / 63 val / 69 test)

### 1.3 训练
- 脚本: `segmentation/scripts/train_seg.py`
- 基础配置: `segmentation/configs/train/seg_full.yaml`
- 预训练: `yolo11s-seg.pt` (COCO 分割)

## 2. 全部实验结果 (12 个实验)

### Round 1 — 架构改进
| 实验 | 模型配置 | Mask mAP50 | Mask mAP50-95 |
|------|---------|-----------|--------------|
| **Baseline** | `bubble_yolo11s_seg.yaml` | 0.836 | **0.529** ★ |
| P3LCRefine+NWD | `bubble_yolo11s_seg_p3lc.yaml` + NWD | **0.847** | 0.508 |
| P3MLCRefine | `bubble_yolo11s_seg_p3mlc.yaml` | 0.836 | 0.519 |
| P3LCRefine+P3SAGate | `bubble_yolo11s_seg_p3lc_sa.yaml` | 0.838 | 0.501 |
| P3+P4+P5 LCRefine | `bubble_yolo11s_seg_p3p4p5_lc.yaml` | 0.840 | 0.513 |

### Round 2 — 训练策略
| 实验 | 配置 | Mask mAP50 | Mask mAP50-95 |
|------|------|-----------|--------------|
| Baseline+NWD | seg_full + --nwd 0.05 | 0.842 | 0.509 |
| Baseline+768 | seg_full_768 (imgsz=768, batch=2) | 0.838 | 0.521 |

### Round 3 — gamma 调优
| 实验 | gamma | Mask mAP50 | Mask mAP50-95 |
|------|-------|-----------|--------------|
| P3LCRefine g=0.01 | 0.01 | 0.835 | 0.513 |
| P3LCRefine g=0.1 | 0.1 | 0.830 | 0.512 |

### Round 4 — SGD 优化器 (来自检测 pipeline 启发)
| 实验 | 配置 | Mask mAP50 | Mask mAP50-95 |
|------|------|-----------|--------------|
| SGD+零增强 | seg_full_sgd_zeroaug | 0.829 | 0.528 |
| SGD+当前增强 | seg_full_sgd_currentaug | 0.822 | 0.519 |
| 检测BB迁移 | det best.pt → seg backbone | 0.834 | 0.499 |

## 3. 关键发现

1. **标准 YOLO11s-seg 架构对本任务已最优** — 任何架构修改都不能同时提升 mAP50 和 mAP50-95
2. **NWD loss 提升 mAP50 但损害 mAP50-95** — 适用于需要更高召回率的场景
3. **SGD + cos_lr + 零增强** — 检测 pipeline 最佳配方在分割上也接近最优 (mAP50-95 0.528 vs 0.529)，收敛速度快 4.6 倍
4. **检测 backbone 迁移无效** — 检测 checkpoint (P3LCRefine+768px+强增强) 的特征与分割模型不兼容
5. **数据增强应保守** — 气泡数据集对 mosaic/mixup/HSV 敏感，零增强最优

## 4. 推荐模型

### 首选: Baseline
- 路径: `segmentation/runs/bubble_seg/bubble_seg_baseline/weights/best.pt`
- Mask mAP50: **0.836**, Mask mAP50-95: **0.529**
- 架构: 标准 YOLO11s-seg, 10M 参数, 33 GFLOPs
- 训练: AdamW, 100 epoch (best@46), batch=4, imgsz=640

### 备选: P3LCRefine+NWD (需要更高 mAP50 时)
- 路径: `segmentation/runs/bubble_seg/bubble_seg_p3lc/weights/best.pt`
- Mask mAP50: **0.847**, Mask mAP50-95: 0.508

## 5. 推理示例

```python
from ultralytics import YOLO

model = YOLO("segmentation/runs/bubble_seg/bubble_seg_baseline/weights/best.pt")
results = model("image.jpg")
for r in results:
    if r.masks is not None:
        # r.masks.xy: polygon coordinates
        # r.boxes: bounding boxes + confidence
        print(f"Found {len(r.masks)} bubbles")
```

## 6. 后续改进方向

按优先级排序:
1. **增加训练数据** — 当前仅 170 张图，更多标注是最有效的提升手段
2. **测试时增强 (TTA)** — 推理时多尺度+翻转融合
3. **更高分辨率训练** — 若 GPU 允许 batch≥4 的 768px 训练
4. **完整 COCO 预训练微调** — 在 COCO 分割上先训练 custom module，再迁移到气泡
