# 03 — 模型结构详解

## Bubble-YOLO11s 架构

Bubble-YOLO11s 基于 YOLO11s 检测框架，在颈部 P3 层与检测头之间插入一个轻量级 P3LCRefine 模块。

### 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Bubble-YOLO11s                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Backbone │───→│   Neck   │───→│  Detect Head     │  │
│  │ (YOLO11s)│    │ (YOLO11s)│    │  P3/P4/P5 heads  │  │
│  │          │    │          │    │                  │  │
│  │  Layer 0 │    │          │    │  P5 (80×80×64)   │  │
│  │    ↓     │    │  ┌─────┐ │    │  P4 (40×40×128)  │  │
│  │  Layer 4 │───→│→│Concat│→│───→│  P3 (20×20×256)  │  │
│  │    ↓     │    │  └─────┘ │    │        ↑         │  │
│  │  Layer 6 │    │    ↑     │    │   ┌──────────┐   │  │
│  │    ↓     │    │  Upsample│    │   │P3LCRefine│   │  │
│  │  Layer 9 │    │    ↑     │    │   │ (128ch)  │   │  │
│  │          │    │  P3 Feat │←───│───│ γ=1.0 init│   │  │
│  └──────────┘    └──────────┘    └───┴──────────┘   │  │
│                                                      │  │
│  Params: 9,414,340   FLOPs: 21.3G   Input: 768×768  │  │
└─────────────────────────────────────────────────────────┘
```

### P3LCRefine 模块

**位置**: 颈部 P3 输出之后，检测头 P3 输入之前（模型第 17 层）

**配置参数**:
- 输入通道: 128
- 深度卷积核: 3×3
- gamma 初始化: 1.0（可学习参数）
- 恒等初始化策略: 深度卷积权重初始化为近零，使模块以近似恒等映射起步

**代码路径**: `ultralytics_custom/bubble_modules.py::P3LCRefine`

**计算流程**:
```
x (B, 128, H, W)
    │
    ├── AvgPool(k=3) → x_avg
    ├── contrast = x - x_avg              # 提取局部对比度
    ├── contrast' = DWConv3×3(contrast)   # 3×3 深度可分离卷积
    ├── out = γ × contrast'               # 可学习缩放
    └── y = x + out                        # 残差连接
```

**设计理念**:
1. **对比度提取** (`x - AvgPool(x)`): 增强局部边缘和纹理
2. **深度可分离卷积**: 最小化参数量（仅 ~1.2K）
3. **残差连接 + 恒等初始化**: 插入预训练网络时不破坏原始特征
4. **仅 P3**: 气泡主要是小目标，P3（stride=8）是最相关的特征层

### NWD Loss 模块

**位置**: `ultralytics_custom/bubble_loss.py`

**计算流程**:
```
预测框 (x₁,y₁,x₂,y₂), 真值框 (x₁ᵍ,y₁ᵍ,x₂ᵍ,y₂ᵍ)
    │
    ├── 建模为 2D 高斯: N(μ, Σ)
    │   μ = (cx, cy), Σ = diag(w²/4, h²/4)
    │
    ├── Wasserstein² = ‖μ₁-μ₂‖² + ‖Σ₁^(1/2)-Σ₂^(1/2)‖_F²
    │
    ├── NWD = exp(-Wasserstein² / C)    # C = 12.8
    │
    └── L_box = (1-0.05)×L_CIoU + 0.05×(1-NWD)
```

### 参数量与计算量

| 组件 | Params | FLOPs |
|------|--------|-------|
| YOLO11s Backbone | ~5.2M | ~12.5G |
| YOLO11s Neck | ~3.5M | ~8.5G |
| YOLO11s Detect | ~0.7M | ~0.2G |
| P3LCRefine | ~1.2K | ~0.01G |
| **Total** | **~9.41M** | **~21.3G** |

P3LCRefine 增加的参数量仅占总参数的 0.013%，几乎可以忽略。

### 权重初始化策略

使用 COCO 预训练的 `yolo11s.pt` 作为初始化。通过 `ultralytics_custom/weight_transfer.py` 中的 `load_bubble_remapped_weights()` 函数，将 YOLO11s 的权重智能映射到修改后的拓扑结构：
- 插入点之前的层: 直接复制
- 插入点之后的层: 索引偏移
- 新增模块（P3LCRefine）: 恒等初始化

### 模型配置文件

`configs/models/bubble_yolo11s_p3_lc_gamma100.yaml`:
```yaml
# YOLO11s backbone (layers 0-8)
# ... standard YOLO11s backbone ...

# Neck with P3LCRefine inserted before P3 detect input
- [-1, 1, P3LCRefine, [128, 3, True, 1.0]]  # P3LC at layer 17

# Detect head (P3, P4, P5)
- [[17, 20, 23], 1, Detect, [nc]]
```
