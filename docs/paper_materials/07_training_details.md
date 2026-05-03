# 07 — 训练配置细节

## 数据集

| 属性 | 值 |
|------|---|
| 数据集名称 | Paper V4 |
| 类别数 | 1 (bubble) |
| 训练集 | 644 张 |
| 验证集 | 83 张 |
| 测试集 | 83 张 |
| 图片尺寸 | 768×768 |
| 数据格式 | YOLO txt (归一化 xywh) |
| 数据 YAML | `yolo_dataset_paper_v4/bubble.yaml` |

## 训练超参数

```yaml
# 最终最佳训练配置
model: configs/models/bubble_yolo11s_p3_lc_gamma100.yaml
data: yolo_dataset_paper_v4/bubble.yaml
pretrained: yolo11s.pt                    # COCO 预训练
imgsz: 768
epochs: 30
batch: 8
device: Tesla V100-SXM2-16GB (单卡)
optimizer: SGD
lr0: 0.001                               # 初始学习率
lrf: 0.01                                # 最终 lr = lr0 × lrf = 1e-5
momentum: 0.937
weight_decay: 0.0005
cos_lr: true                             # 余弦退火
warmup_epochs: 3.0                       # 线性预热
warmup_momentum: 0.8
warmup_bias_lr: 0.1
close_mosaic: 10                         # 最后 10 epoch 关闭 Mosaic/MixUp
amp: true                                # 自动混合精度

# 数据增强
mosaic: 1.0                              # 100% Mosaic（前 20 epoch）
mixup: 0.2                               # 20% MixUp
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.4
degrees: 0.0
translate: 0.02
scale: 0.05
fliplr: 0.5
flipud: 0.0

# NWD Loss
use_nwd: true
nwd_weight: 0.05                         # CIoU/NWD 融合权重
nwd_constant: 12.8

# 评估
selector_metric: map50_balanced
selector_eval_mode: online               # 从 results.csv 选最佳 epoch
selector_precision_min: 0.75
selector_recall_min: 0.72
conf_sweep: [0.10, 0.15, 0.25, 0.35]
```

## 最佳权重选择策略

`train_experiment.py` 使用两阶段选择：
1. 在验证集上评估所有 epoch 的 best.pt / last.pt / epoch*.pt
2. 筛选 precision ≥ 0.75 且 recall ≥ 0.72 的候选
3. 选择 mAP50 最高的作为 `map50_selected.pt`

## 训练曲线特征

基于最终最佳模型 (seed48) 的 `results.csv`：

| 指标 | 值 |
|------|---|
| 最佳 epoch | 18 |
| 最终 epoch | 30 |
| 最佳 val mAP50 | 0.828 |
| 最终 val mAP50 | 0.822 |
| mAP50 下降 | 0.006 |
| 异常值 (NaN/Inf) | 0 |
| 训练耗时 | ~4.6 分钟 / epoch |
| 总训练时间 | ~2.3 小时 (30 epoch) |

## 硬件配置

| 组件 | 规格 |
|------|------|
| GPU | Tesla V100-SXM2-16GB × 2（仅用单卡） |
| CPU | Intel Xeon |
| RAM | 16 GB |
| OS | Ubuntu Linux |
| Python | 3.11.14 |
| PyTorch | 2.4.0+cu124 |
| Ultralytics | 8.3.227 |
