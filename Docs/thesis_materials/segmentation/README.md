# Bubble-YOLO11s-seg 分割论文材料

> 气泡实例分割毕业设计论文的完整材料库 — SAM3 → 数据集构建 → 模型训练 → 消融实验

## 快速导航

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| [01_SAM3标注生成.md](01_SAM3标注生成.md) | SAM3 零样本分割标注管线 | 第 3 章「分割标注获取」的材料 |
| [02_数据集构建.md](02_数据集构建.md) | COCO → YOLO-seg 数据集构建流程 | 第 3 章「分割数据集」的材料 |
| [03_训练方法论.md](03_训练方法论.md) | YOLO11s-seg 架构、训练策略、Dice Loss 改进 | 第 4 章「分割改进方法」的材料 |
| [04_实验设计与结果.md](04_实验设计与结果.md) | 17 个消融实验、Dice sweep、最终排名 | 第 5 章「分割实验结果」的材料 |
| [05_讨论与分析.md](05_讨论与分析.md) | 架构改动为何失败、SGD vs AdamW、Dice weight 敏感性 | 第 6 章「讨论」的材料 |
| [06_训练配置细节.md](06_训练配置细节.md) | 完整超参数、模型配置、环境信息 | 实验复现与附录 |

### 表格

| 文件 | 内容 |
|------|------|
| [tables/ablation_table.md](tables/ablation_table.md) | 完整消融实验表、Dice weight sweep、优化器对比 |
| [tables/main_results_table.md](tables/main_results_table.md) | 分割 vs 检测对比、指标分解、最终模型性能 |
| [tables/improvement_journey.md](tables/improvement_journey.md) | 从 baseline 到最终模型的 6 轮改进历程 |

---

## 核心数据速览

Bubble-YOLO11s-seg 在标准 YOLO11s-seg 基线上实现了**同时提升 mAP50 和 mAP50-95**：

| 指标 | Baseline (AdamW) | Dice=0.1 (SGD) | 提升 |
|------|-----------------|----------------|------|
| **Mask mAP50** | 0.836 | **0.839** | +0.003 |
| **Mask mAP50-95** | 0.529 | **0.538** | +0.009 |
| Box mAP50 | 0.845 | 0.837 | -0.008 |
| Box mAP50-95 | 0.532 | 0.542 | +0.010 |
| 收敛速度 (best epoch) | 46 | **12** | 3.8× faster |

### 数据集规模
- 原始: 170 张光学流体图像, 9,524 个 bbox → SAM3 生成 9,504 个分割标注 (99.79%)
- 训练: 712 张 640×640 瓦片, 验证: 63 张, 测试: 69 张

### 关键发现
1. **Dice Loss (w=0.1) 是关键突破**: mAP50-95 从 0.529 → 0.538
2. **SGD 优于 AdamW**: 小数据集上泛化更好, 收敛快 4.6×
3. **零增强最优**: 气泡形态对 mosaic/mixup/HSV 敏感
4. **17 个架构改进实验全部失败**: 包括 P3LCRefine、P3MLCRefine、SPD-Conv、P3SAGate 等

## 数据来源

- SAM3 推理: `segmentation/generate_masks.py`
- 数据集构建: `segmentation/build_dataset.py`
- 训练脚本: `segmentation/scripts/train_seg.py`
- Dice Loss: `segmentation/scripts/dice_loss.py`
- 模型配置: `segmentation/configs/models/`
- 训练配置: `segmentation/configs/train/`
- 实验总结: `segmentation/EXPERIMENT_SUMMARY.md`
- 改进探索: `segmentation/IMPROVEMENT_EXPLORATION.md`
- 研究上下文: `segmentation/RESEARCH_CONTEXT.md`
