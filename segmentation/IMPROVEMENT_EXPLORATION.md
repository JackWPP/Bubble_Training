# 气泡分割改进空间 — 全面探索

> 当前最佳: Dice Loss (SGD+zero aug), Mask mAP50=0.828, mAP50-95=0.532
> 低于检测最佳 (mAP50=0.889, mAP50-95=0.592) 约 0.06 点 — 仍有提升空间

## 一、已尝试方法总结 (13 实验)

| 类别 | 方法 | 效果 |
|------|------|------|
| 架构 | P3LCRefine, P3MLCRefine, P3SAGate, P3+P4+P5 LCRefine | 均未超越 baseline |
| 损失 | NWD loss (weight=0.05) | mAP50↑, mAP50-95↓ |
| 损失 | **Dice loss (weight=0.3)** | mAP50-95 **+0.003** ★ |
| 优化器 | SGD + cos_lr (替换 AdamW) | 收敛 4.6x 快, 精度持平 |
| 增强 | 零增强 (替换 mosaic=0.3) | 配合 SGD 效果最优 |
| 分辨率 | 768px (batch=2) | batch 太小, 未体现收益 |
| 预训练 | 检测 backbone 迁移 | 不兼容 |
| 架构 | P2 head | parse_model 通道计算 bug |
| 架构 | SPD-Conv | 精度低于 baseline |

## 二、已验证的关键 insight

1. **Dice loss 有效** — 直接优化 mask IoU，解决 BCE 对边界像素梯度不足问题
2. **SGD > AdamW** — 小数据集上泛化更好
3. **零增强最优** — 气泡形态对空间扭曲敏感
4. **架构改动无效** — 16 个自定义模块在分割上均无法同时提升 mAP50 和 mAP50-95

## 三、未探索的改进方向

### A. Dice Loss 调优（低风险，立即可做）

Dice weight=0.3 是随意选的，存在优化空间：

| Dice weight | 含义 |
|-------------|------|
| 0.1 | 轻度 Dice，BCE 主导 |
| 0.3 | 当前值 |
| 0.5 | 等权混合 |
| 0.7 | Dice 主导，BCE 辅助 |

**建议**: Dice weight sweep [0.1, 0.3, 0.5, 0.7]，每个 ~25 分钟。可能找到更好的平衡点。

### B. Dice + NWD 组合（低风险）

Dice 改善边界精度的同时，NWD 改善小目标召回。两者组合可能互补：
- Dice: 提升 mAP50-95（边界精度）
- NWD: 提升 mAP50（小目标召回）
- 预期: 同时提升两个指标

**命令**: `--dice 0.3 --nwd 0.05`

### C. Dice + 768px 分辨率（中风险）

SGD + batch=2 at 768px 之前因 batch 太小效果不佳。但可以尝试梯度累积模拟 batch=4：
- 实际 batch=2, accumulate=2 → 等效 batch=4
- 加上 Dice loss

### D. EMA 权重（低风险，配置级）

Exponential Moving Average 在分割任务中常带来 0.5-1.0 点提升。Ultralytics 不支持原生 EMA，但可通过自定义 callback 实现或在推理时手动平均最后 N 个 checkpoint。

### E. 训练更久 + 更低最终 LR（低风险）

当前 lrf=0.01 意味着最终 LR=1e-5。检测实验最佳收敛在 epoch 10-15，我们的 Dice 在 epoch 13 达到最佳。如果 lrf=0.001 (最终 LR=1e-6)，可能给模型更多精细调优空间。

### F. Dice + Baseline AdamW 配置（低风险）

Dice loss 只在 SGD+zero aug 配置下测试过。在原始 AdamW+mosaic=0.3 配置下测试 Dice 可能产生不同效果 — 更强的增强可能需要更强的边界损失。

### G. Boundary Loss（中风险，需代码实现）

研究报告强烈推荐的 Surface/Boundary Loss：计算 GT mask 的距离变换图，对远离边界的错误施加指数级惩罚。需要：
1. 离线预计算所有训练样本的距离变换图
2. 在 `single_mask_loss` 中加载并计算 boundary loss
3. 与 Dice+BCE 混合

**实现难度**: 中等（~100 行代码 + 数据预处理）

### H. 增加 Proto 掩码数量（中风险）

当前 proto=32, hidden=256。增加到 proto=64 或 hidden=512 可能提升掩码表达能力。代价是 ~10% 参数增加。

### I. 数据集改进（高风险高回报）

1. **SAM3 mask 质量**: 当前 polygon ≤50 顶点。尝试 ≤100 顶点或保留原始 mask，可能改善边界标注质量
2. **减少瓦片化噪声**: stride=640 (无重叠) 或 stride=560 (10% 重叠)
3. **增加数据**: 170 张图对分割可能不够。可考虑:
   - 回译增强 (copy-paste 气泡到新背景)
   - 使用 SAM3 在新未标注图像上生成伪标签

### J. 推理端改进（零风险，不改训练）

1. **TTA (测试时增强)**: 推理时水平翻转 + 多尺度 (0.83x, 1.0x, 1.17x)，融合结果
2. **SAHI 推理**: 对原始大图进行重叠瓦片推理 + NMS 融合，避免瓦片边界切分气泡

## 四、优先级排序

| 优先级 | 方向 | 预期收益 | 风险 | 时间 |
|--------|------|---------|------|------|
| **P0** | Dice weight sweep | +0.002-0.005 mAP50-95 | 低 | 1h |
| **P0** | Dice + NWD 组合 | 同时提升双指标 | 低 | 25m |
| **P1** | Dice + AdamW 配置 | 可能发现更好的增强-损失组合 | 低 | 25m |
| **P1** | EMA / 更长训练 | +0.003-0.005 | 低 | 30m |
| **P2** | 768px + 梯度累积 | +0.005-0.010 | 中 | 30m |
| **P2** | Boundary Loss | +0.005-0.010 | 中 | 2h |
| **P3** | 数据集改进 | +0.010-0.030 | 高 | 数小时 |
| **P3** | P2 head (修 bug) | +0.010-0.020 | 高 | 未知 |
