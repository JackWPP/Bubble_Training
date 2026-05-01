# 气泡检测 YOLO 数据集构建报告

## 构建概览

本数据集由 `Dataset` 目录中的 COCO 导出合并生成，所有标注统一映射为单类 `bubble`，YOLO 类别编号为 `0`。

- 输出目录：`G:/Bubble_Train/yolo_dataset_balanced_v3`
- 训练配置：`G:/Bubble_Train/yolo_dataset_balanced_v3/bubble.yaml`
- split 模式：`balanced-v3`
- 随机种子：`42`
- 目标比例：train/val/test = 0.6/0.2/0.2

`group` 模式会先按物理源头或采集条件生成 `group_key`，再做 train/val/test 划分。这样同一视频、同一实验条件或同一采集序列不会同时进入训练集和验证/测试集，可作为论文最终泛化测试集。

## 原始来源统计

| 来源 | 有效图像数 | 有效标注框数 | 缺失图像数 | 无效框数 | 主要分辨率 |
| --- | ---: | ---: | ---: | ---: | --- |
| 20+40 | 66 | 2233 | 0 | 0 | 502x404(34), 502x400(32) |
| 60+80 | 47 | 442 | 0 | 0 | 508x426(24), 504x424(23) |
| big_fengchao | 6 | 1460 | 0 | 0 | 1920x1080(6) |
| bubble_1 | 3 | 590 | 0 | 0 | 1920x1080(3) |
| bubble_fc | 12 | 602 | 0 | 0 | 480x480(11), 1080x1080(1) |
| bubble_pad | 3 | 1484 | 0 | 0 | 1920x1080(3) |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0 | 33 | 2713 | 0 | 0 | 1920x1080(33) |

原始类别统计：`{'bubble': 9524}`

## 处理策略

- 大图使用 `640x640` 滑动窗口切片，stride 为 `480`，边缘自动补最后一窗。
- 小图保持比例缩放，并居中 padding 到 `640x640`。
- 切片内 bbox 需要满足：中心点落入 tile；裁剪后宽高不少于 `4px`；保留面积不少于原框面积的 `40%`。
- 离线增强只作用于 `train`，`val` 和 `test` 保持未增强。
- `manifest.json` 同时记录 `source_key` 和 `group_key`，用于复查精确原图泄漏与场景级泄漏。

## 输出统计

| Split | 图像数 | 标签文件数 | 标注框数 | 基础样本数 | 增强样本数 | 平均框/图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1470 | 1470 | 40785 | 294 | 1176 | 27.74 |
| val | 97 | 97 | 3007 | 97 | 0 | 31.00 |
| test | 97 | 97 | 2792 | 97 | 0 | 28.78 |

bbox 宽度统计：`{'count': 46584, 'min': 6.31, 'p50': 45.7, 'p90': 75.88, 'max': 312.4, 'mean': 47.332}`

bbox 高度统计：`{'count': 46584, 'min': 5.87, 'p50': 44.111, 'p90': 74.03, 'max': 305.81, 'mean': 46.283}`

## 泄漏与质量检查

- 精确原图 source_key 跨 split 数：`0`
- 物理源头 group_key 跨 split 数：`12`
- val/test 离线增强样本数：`0`
- 校验结果：`PASS`，错误数：`0`

## 来源覆盖

| 来源 | train 图像数 | val 图像数 | test 图像数 | 总图像数 | 总标注框数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20+40 | 200 | 13 | 13 | 226 | 7913 |
| 60+80 | 145 | 9 | 9 | 163 | 1454 |
| big_fengchao | 160 | 8 | 8 | 176 | 9805 |
| bubble_1 | 40 | 8 | 8 | 56 | 2678 |
| bubble_fc | 45 | 3 | 3 | 51 | 1927 |
| bubble_pad | 40 | 8 | 8 | 56 | 6398 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0 | 840 | 48 | 48 | 936 | 16409 |

## Group 覆盖

| group_key | train 图像数 | val 图像数 | test 图像数 | 总图像数 | 总标注框数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20+40|capture_series | 200 | 13 | 13 | 226 | 7913 |
| 60+80|capture_series | 145 | 9 | 9 | 163 | 1454 |
| big_fengchao|video_or_scene | 160 | 8 | 8 | 176 | 9805 |
| bubble_1|video_or_scene | 40 | 8 | 8 | 56 | 2678 |
| bubble_fc|-40-0.16A-1(480P).mp4 | 5 | 1 | 1 | 7 | 318 |
| bubble_fc|-40-0.2A-1(480p).mp4 | 10 | 1 | 1 | 12 | 586 |
| bubble_fc|-40-0.2A-1.mp4 | 20 | 0 | 0 | 20 | 425 |
| bubble_fc|-40-0.4A-1(480P).mp4 | 10 | 1 | 1 | 12 | 598 |
| bubble_pad|video_or_scene | 40 | 8 | 8 | 56 | 6398 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.04-1 | 240 | 16 | 16 | 272 | 5825 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.08-2 | 240 | 16 | 16 | 272 | 2599 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.12-1 | 160 | 8 | 8 | 176 | 4225 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.16-2 | 200 | 8 | 8 | 216 | 3760 |

## 论文使用建议

`yolo_dataset_integrated` 可作为历史随机原图划分与快速开发集；`yolo_dataset_grouped` 应作为正式泛化测试集。论文中建议表述为：本文按物理源头和采集条件隔离训练、验证与测试数据，避免相邻帧、同实验条件和同源切片导致的指标虚高。

## Balanced V3 Notes

- split target: train/val/test = 0.60/0.20/0.20
- augmentation profile: `uniform`
- source_key leakage is forbidden; group leakage is expected for this main-training split.
- Primary detector metrics for this phase are mAP@50, precision, and recall; mAP@50-95 is kept as a localization diagnostic.
