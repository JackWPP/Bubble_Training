# 最佳权重清单

## 论文推荐使用的权重

### 主结果 (seed44, 论文使用)
```
runs/bubble_paper_v4_coco_aug/PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed44/weights/map50_selected.pt
```
- main mAP50: 0.81712
- main mAP50-95: 0.43842
- main sum: 1.25554

### 最佳结果 (seed48, 附录使用)
```
runs/bubble_paper_v4_coco_aug/PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed48/weights/map50_selected.pt
```
- main mAP50: 0.82535
- main mAP50-95: 0.43765
- main sum: 1.26300

### Baseline (对比)
```
runs/bubble_paper_v4_coco_init/PV4C1_P3LC_COCO_NWD_W005_E30_seed44/weights/map50_selected.pt
```
- main mAP50: 0.81210
- main mAP50-95: 0.42752
- main sum: 1.23962

### Baseline no-NWD (P3LCRefine only)
```
runs/bubble_paper_v4_coco_init/PV4C1_P3LC_COCO_G100_E30/weights/map50_selected.pt
```
- main mAP50: 0.80712
- main mAP50-95: 0.42422
- main sum: 1.23134

### Pure baseline (YOLO11s, no modifications)
```
runs/bubble_paper_v4_yolo11s/PV4S_768_LR0010_yolo11s_paper_v4/weights/map50_selected.pt
```
- main mAP50: 0.79803
- main mAP50-95: 0.41704
- main sum: 1.21507
