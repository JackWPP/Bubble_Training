# Bubble-YOLO11s 毕业论文材料

> 气泡检测毕业设计论文的完整材料库 —— 包含术语规范、论文大纲、方法论、实验数据、架构图和表格

## 快速导航

### 论文写作

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| [00_术语标准化.md](00_术语标准化.md) | 工程用语 → 规范学术术语对照表 | **写论文前必读**，确保全文术语统一 |
| [01_论文大纲.md](01_论文大纲.md) | 论文章节结构、材料映射与待补内容 | 搭建论文框架时参考 |
| [02_数据集构建.md](02_数据集构建.md) | 三代数据集演进叙事 (V1→V2→V3) | 第 3 章「数据集构建」的材料 |
| [03_方法论.md](03_方法论.md) | 三项改进的动机、设计与公式化 | 第 4 章「改进方法」的材料 |
| [04_实验设计与结果.md](04_实验设计与结果.md) | 消融实验、主结果、OOD 泛化 | 第 5 章「实验结果」的材料 |
| [05_讨论与失败分析.md](05_讨论与失败分析.md) | 6 个负结果、种子稳定性、增强饱和 | 第 6 章「讨论」的材料 |
| [06_训练配置细节.md](06_训练配置细节.md) | 完整超参数、权重选择策略、环境 | 实验复现与附录 |

### 图表

| 文件 | 内容 | 用途 |
|------|------|------|
| [figures/arch_overall_v1.svg](figures/arch_overall_v1.svg) | 整体架构 — 简洁学术风 | 论文图 1: Bubble-YOLO11s 结构概览 |
| [figures/arch_overall_v2.svg](figures/arch_overall_v2.svg) | 整体架构 — 详细标注风 (含层索引、P3LCRefine 内部) | 论文图 1 备选/详细版 |
| [figures/p3lcrefine_v1.svg](figures/p3lcrefine_v1.svg) | P3LCRefine 模块 — 数据流风 | 论文图 2: P3LCRefine 内部结构 |
| [figures/p3lcrefine_v2.svg](figures/p3lcrefine_v2.svg) | P3LCRefine 模块 — 张量形状标注风 | 论文图 2 备选/详细版 |
| [figures/nwd_loss_v1.svg](figures/nwd_loss_v1.svg) | NWD 损失 — 流程风 | 论文图 3: NWD 损失融合流程 |
| [figures/nwd_loss_v2.svg](figures/nwd_loss_v2.svg) | NWD 损失 — 数学公式风 (含消融表) | 论文图 3 备选/详细版 |
| [figures/improvement_journey.svg](figures/improvement_journey.svg) | 改进历程时间线 | 论文图 4: 从基线到最终模型的提升轨迹 |
| [figures/training_pipeline.svg](figures/training_pipeline.svg) | 训练与评估完整流程 | 论文图 5: 实验流程 |
| [figures/mosaic_illustration.svg](figures/mosaic_illustration.svg) | Mosaic 增强示意图 | 论文图 6: 数据增强原理 |
| [figures/training_curves.png](figures/training_curves.png) | 训练曲线 | 附录材料 |
| [figures/augmentation_saturation.png](figures/augmentation_saturation.png) | 增强饱和曲线 | 讨论部分 |

### 表格

| 文件 | 内容 |
|------|------|
| [tables/ablation_table.md](tables/ablation_table.md) | 组件消融、NWD 超参数消融、增强消融、负结果汇总 |
| [tables/main_results_table.md](tables/main_results_table.md) | 主结果对比、指标分解、推理效率、改进历程 |
| [tables/ood_results_table.md](tables/ood_results_table.md) | OOD 结果、增强影响、测试集构成、种子稳定性 |
| [weights_reference.md](weights_reference.md) | 五组权重路径与完整指标 |

---

## 核心数据速览

| 指标 | 基线 (YOLO11s) | Bubble-YOLO11s (seed44) ★ 论文 | Bubble-YOLO11s (seed48) ☆ 最佳 |
|------|---------------|-------------------------------|-------------------------------|
| 主测试集 mAP@50 | 0.79803 | 0.81712 | 0.82535 |
| 主测试集 mAP@50:95 | 0.41704 | 0.43842 | 0.43765 |
| 综合指标 S | 1.21507 | 1.25554 (+3.33%) | 1.26300 (+3.94%) |
| OOD mAP@50 | 0.87396 | 0.85368 | 0.86199 |
| OOD mAP@50:95 | 0.55276 | 0.53439 | 0.53569 |
| 参数量 | 9.41M | 9.41M | 9.41M |
| FLOPs | 21.3G | 21.3G | 21.3G |

> ★ 论文推荐使用 seed44 作为主结果（标准流程，无选择偏差），seed48 作为附录补充。

## 三项改进

1. **局部对比度细化模块 (P3LCRefine)** — 约 1,280 参数，插入 Neck P3 层与检测头之间，通过 `x + γ × DWConv(x − AvgPool(x))` 增强小目标局部边缘纹理
2. **归一化 Wasserstein 距离损失融合 (NWD Loss)** — 将边框建模为二维高斯分布，以 5:95 比例与 CIoU 融合，提供平滑的小目标定位梯度
3. **强数据增强配方** — Mosaic (p=1.0) + HSV 扰动 + MixUp (p=0.2)，在 644 张训练图上最大化数据多样性

## 数据来源

- 实验指标: `runs/summary/paper_v4_all_experiment_summary_latest.csv`
- 训练配置: `configs/train/paper_v4_coco_init_lr0010_e30_mosaic10_hsv_mixup.yaml`
- 模型定义: `configs/models/bubble_yolo11s_p3_lc_gamma100.yaml`
- 自定义模块: `ultralytics_custom/bubble_modules.py`
- NWD 损失: `ultralytics_custom/bubble_loss.py`
- 权重重映射: `ultralytics_custom/weight_transfer.py`
- 论文原始材料: `Docs/paper_materials/` (10 章实验笔记)
- 原始数据集报告: `Docs/DATASET_BUILD_REPORT.md`, `Docs/DATASET_GROUPED_BUILD_REPORT.md`

## 使用说明

1. **写论文前**: 先读 `00_术语标准化.md`，了解每个概念在论文中的正确表述
2. **搭框架**: 参考 `01_论文大纲.md`，了解每章应涵盖的内容和已有材料
3. **写正文**: 各章节的材料文件（02-06）可直接提取内容用于论文写作
4. **插图表**: SVG 图可直接嵌入论文（LaTeX 或 Word），或导出为 PDF/PNG 格式
5. **填数据**: `tables/` 中的表格可直接复制使用
