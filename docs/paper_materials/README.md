# Bubble-YOLO11s 论文材料

> 气泡检测毕业设计论文的完整实验材料、分析文档和图表

## 目录

| 文件 | 内容 | 论文对应 |
|------|------|---------|
| [01_experiment_chronicle.md](01_experiment_chronicle.md) | 18 个实验的完整编年史 | 实验设计 |
| [02_methodology.md](02_methodology.md) | 三项改进的系统方法论 | 方法章节 |
| [03_model_architecture.md](03_model_architecture.md) | Bubble-YOLO11s 模型结构 | 模型架构 |
| [04_ablation_study.md](04_ablation_study.md) | 组件消融、NWD 消融、增强消融 | 消融研究 |
| [05_main_results.md](05_main_results.md) | 最终主结果表与对比 | 实验结果 |
| [06_ood_generalization.md](06_ood_generalization.md) | OOD 泛化性能分析 | 泛化性 |
| [07_training_details.md](07_training_details.md) | 完整训练超参数与配置 | 实现细节 |
| [08_augmentation_analysis.md](08_augmentation_analysis.md) | 7 种增强方法的系统分析 | 数据增强 |
| [09_failure_analysis.md](09_failure_analysis.md) | 6 个负结果的教训总结 | 讨论 |
| [10_seed_stability.md](10_seed_stability.md) | 种子稳定性与可重复性 | 可重复性 |
| [figures/](figures/) | 架构图、训练曲线、提升路径 | 图表 |

## 核心数据速览

| 指标 | Baseline | Bubble-YOLO11s (seed44) | Bubble-YOLO11s (seed48) |
|------|----------|------------------------|------------------------|
| main mAP50 | 0.79803 | 0.81712 | **0.82535** |
| main mAP50-95 | 0.41704 | 0.43842 | **0.43765** |
| main sum | 1.21507 | 1.25554 | **1.26300** |
| vs baseline | — | +3.33% | **+3.94%** |
| 参数量 | 9.41M | 9.41M | 9.41M |
| FLOPs | 21.3G | 21.3G | 21.3G |

## 最佳权重

```
runs/bubble_paper_v4_coco_aug/PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed48/weights/map50_selected.pt
```

## 数据来源

- 实验数据: `runs/summary/paper_v4_all_experiment_summary_latest.csv`
- 训练配置: `configs/train/paper_v4_coco_init_lr0010_e30_mosaic10_hsv_mixup.yaml`
- 模型配置: `configs/models/bubble_yolo11s_p3_lc_gamma100.yaml`
- 自定义模块: `ultralytics_custom/bubble_modules.py`
- Loss 实现: `ultralytics_custom/bubble_loss.py`
