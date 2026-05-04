# 权重文件索引

## 论文推荐使用的权重

### 主结果 (seed=48, 论文使用)

```
runs/bubble_paper_v4_coco_aug/PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed48/weights/map50_selected.pt
```

| 指标 | 值 |
|------|-----|
| 主测试集 mAP@50 | 0.82535 |
| 主测试集 mAP@50:95 | 0.43765 |
| 主测试集综合指标 S | 1.26300 |
| OOD mAP@50 | 0.86199 |
| OOD mAP@50:95 | 0.53569 |
| 参数量 | 9,415,621 |
| FLOPs | ~21.3G |

### 参考结果 (seed=44)

```
runs/bubble_paper_v4_coco_aug/PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed44/weights/map50_selected.pt
```

| 指标 | 值 |
|------|-----|
| 主测试集 mAP@50 | 0.81712 |
| 主测试集 mAP@50:95 | 0.43842 |
| 主测试集综合指标 S | 1.25554 |
| OOD mAP@50 | 0.85368 |
| OOD mAP@50:95 | 0.53439 |

---

## 对照权重

### 内部基线：P3LCRefine + NWD (无 Mosaic 增强, seed=44)

> 注：此权重文件路径待确认。指标数据来源为 `runs/summary/paper_v4_all_experiment_summary_latest.csv`。
>
> 预期路径：
> ```
> runs/bubble_paper_v4_coco_init/PV4C1_P3LC_COCO_NWD_W005_E30_seed44/weights/map50_selected.pt
> ```

| 指标 | 值 |
|------|-----|
| 主测试集 mAP@50 | 0.81210 |
| 主测试集 mAP@50:95 | 0.42752 |
| 主测试集综合指标 S | 1.23962 |

### 内部基线：仅 P3LCRefine (无 NWD, 无增强)

```
runs/bubble_paper_v4_coco_init/PV4C1_P3LC_COCO_G100_E30/weights/map50_selected.pt
```

| 指标 | 值 |
|------|-----|
| 主测试集 mAP@50 | 0.80712 |
| 主测试集 mAP@50:95 | 0.42422 |
| 主测试集综合指标 S | 1.23134 |

### 纯基线：YOLO11s (无任何改进)

```
runs/bubble_paper_v4/PV4S_768_LR0010_yolo11s_paper_v4/weights/map50_selected.pt
```

| 指标 | 值 |
|------|-----|
| 主测试集 mAP@50 | 0.79803 |
| 主测试集 mAP@50:95 | 0.41704 |
| 主测试集综合指标 S | 1.21507 |

---

## 预训练权重

| 文件 | 说明 |
|------|------|
| yolo11s.pt (已下载至项目根目录) | YOLO11s MS COCO 预训练权重 |
| configs/models/bubble_yolo11s_p3_lc_gamma100.yaml | 模型结构定义文件 |

## 权重选择说明

所有 `map50_selected.pt` 均经过双阈值筛选（精确率 ≥ 0.75, 召回率 ≥ 0.72），在 train-dev split 上选择 mAP@50 最优的 epoch。该 split 与主测试集和 OOD 测试集均无泄漏。
