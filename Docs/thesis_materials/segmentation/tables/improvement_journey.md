# 分割模型改进历程

## 从 Baseline 到最终模型的 6 轮迭代

```
Round 1 (架构)    Round 2 (训练)    Round 3 (gamma)    Round 4 (SGD)     Round 5 (Dice)    Round 6 (精调)
    │                  │                  │                  │                  │                  │
    ├─ P3LCRefine      ├─ NWD loss        ├─ γ=0.01          ├─ SGD+零增强     ├─ Dice w=0.3     ├─ Dice+NWD
    ├─ P3MLCRefine     └─ 768px           └─ γ=0.1           ├─ SGD+当前增强   ├─ Dice w=0.1 ★  ├─ Dice+AdamW
    ├─ P3SAGate                                              └─ 检测BB迁移     ├─ Dice w=0.5     └─ 768+Dice
    └─ P3P4P5 LC                                                               └─ Dice w=0.05
         │                  │                  │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼                  ▼                  ▼
    全部失败            NWD损害            无效             收敛4.6x快         mAP50-95+0.009    无进一步改善
    mAP50-95↓         mAP50-95↓                          追平baseline       双指标提升
```

## 关键转折点

| 轮次 | 事件 | 影响 |
|:---:|------|------|
| R1 | 所有架构改进失败 | 确立"检测模块≠分割模块"原则 |
| R2 | NWD 的 tradeoff 被确认 | 放弃 NWD 在分割上的应用 |
| R4 | SGD+cos_lr 迁移成功 | 收敛速度提升 4.6×, 打开快速迭代通道 |
| R5 | **Dice w=0.1 突破** | mAP50-95 从 0.528 → 0.538, 同时 mAP50↑ |

## 最终方案

**模型**: YOLO11s-seg (标准架构, 10M 参数)
**训练**: SGD + cos_lr + 零增强 + Dice Loss (w=0.1)
**性能**: Mask mAP50=0.839, Mask mAP50-95=0.538

## 教训

1. **不要堆砌模块**: 17 个架构实验全部失败，最简单的最优
2. **优化器比架构重要**: SGD > AdamW 在这类小数据集分割任务上
3. **损失函数是突破口**: Dice Loss 零参数改进 > 任何架构改动
4. **检测 ≠ 分割**: 检测成功的模块/策略不能直接迁移到分割
