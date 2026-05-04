# 气泡实例分割 — 研究现状与未解决问题

> 本文档面向具备深度调研能力的 Agent，旨在清晰呈现当前研究状态、已尝试的方法、遇到的瓶颈，以及尚未验证的假设。请基于此文档进行搜索调研，提出新的改进方向。

---

## 一、任务定义

**目标**: 对光学流体图像中的气泡进行实例分割（instance segmentation）。

**单类别**: 仅 `bubble` 一个类别。

**评估指标**: Mask mAP50、Mask mAP50-95（COCO 标准）。

---

## 二、数据集

### 来源
7 个 COCO 格式数据源，来自不同实验条件和成像设备：

| 数据源 | 图片数 | 标注框数 | SAM3 分割标注 |
|--------|--------|---------|--------------|
| 20+40 | 66 | 2,233 | 2,233 |
| 60+80 | 47 | 442 | 442 |
| big_fengchao | 6 | 1,460 | 1,451 |
| bubble_1 | 3 | 590 | 589 |
| bubble_fc | 12 | 602 | 602 |
| bubble_pad | 3 | 1,484 | 1,475 |
| job_13 | 33 | 2,713 | 2,712 |
| **合计** | **170** | **9,524** | **9,504 (99.79%)** |

### 标注来源
- 原始标注: COCO bbox-only（人工标注的矩形框）
- 分割标注: SAM3（facebook/sam3，HuggingFace 上约 3.4GB 权重）以 bbox 为 prompt 自动生成 polygonal mask
- SAM3 采用多阶段策略：concept 模式（20 个 bbox 同时输入）+ batch fallback（每组 20 个）+ 单框 fallback

### 数据集构建
- 脚本: `build_dataset.py`
- 策略: 图片按 source_key 分层分割为 70/15/15 (train/val/test)
- 处理: 大图 (>640px) 滑窗瓦片化 (stride=480)，小图 letterbox 到 640×640
- 增强: 仅训练集水平翻转
- 输出: 712 train / 63 val / 69 test 张 640×640 瓦片

### 气泡特征
- 气泡是**小目标**（在 640px 图像中通常占 10-80px 区域）
- 形态多样：圆形、椭圆形、不规则形
- 部分透明，边界模糊
- 密集分布（部分图片有 400-500 个气泡）
- 存在重叠和遮挡

---

## 三、当前最佳模型

### 模型架构
**YOLO11s-seg**（标准 Ultralytics 实现）:
- Backbone: YOLO11s (Conv + C3k2 + SPPF + C2PSA), 10 层
- Neck: FPN 风格，P3/P4/P5 三尺度输出
- Head: Segment head，32 个原型掩码，256 隐藏通道
- 参数: ~10M，FLOPs: ~33G
- 预训练: `yolo11s-seg.pt` (COCO 分割)

### 训练配置
```yaml
optimizer: AdamW
lr0: 0.001
epochs: 100 (best@46, early stop@76)
batch: 4
imgsz: 640
augmentations: mosaic=0.3, mixup=0.1, hsv_h=0.01, hsv_s=0.3, hsv_v=0.2, fliplr=0.5
pretrained: yolo11s-seg.pt (COCO segmentation)
```

### 最佳结果
| 指标 | 值 |
|------|-----|
| Box mAP50 | 0.845 |
| Box mAP50-95 | 0.532 |
| **Mask mAP50** | **0.836** |
| **Mask mAP50-95** | **0.529** |

---

## 四、已尝试的改进方法及结果

### 第 1 轮: 架构改进（基于项目自有 16 个自定义模块）

项目 `ultralytics_custom/bubble_modules.py` 已有 16 个自定义 PyTorch 模块，来自检测实验的消融研究。我们将其中最有潜力的几个移植到分割模型。

#### 1.1 P3LCRefine（局部对比度细化）
- **原理**: 计算 `x - avg_pool(x)` 得到局部对比度信号，通过 depthwise conv 增强。gamma_init=1.0 表示残差路径初始权重为 1.0
- **位置**: P3 特征层（最高分辨率，负责小目标）
- **结果**: Mask mAP50=0.847 (+0.011), mAP50-95=0.508 (-0.021)
- **分析**: mAP50 提升说明检测到更多气泡，但 mAP50-95 下降说明边界精度变差。P3 特征被过度处理，破坏了 COCO 预训练特征

#### 1.2 P3MLCRefine（多尺度局部对比度）
- **原理**: 同时用 3×3 和 5×5 两个核计算对比度，融合后输出。理论上能捕获不同尺度的边界
- **结果**: Mask mAP50=0.836 (持平), mAP50-95=0.519 (-0.010)
- **分析**: 多尺度对比度没有带来额外收益

#### 1.3 P3LCRefine + P3SAGate（边界增强 + 空间注意力）
- **原理**: P3LCRefine 后接空间注意力门控，让网络聚焦边界位置。灵感来自 TinySeg 论文
- **结果**: Mask mAP50=0.838, mAP50-95=0.501 (-0.028)
- **分析**: 注意力门控反而损害了性能

#### 1.4 P3+P4+P5 全部放置 LCRefine
- **原理**: 仿照检测最佳模型（SSBRefine+GLRB at P3/P4/P5），在所有三个输出尺度放置 LCRefine
- **结果**: Mask mAP50=0.840, mAP50-95=0.513 (-0.016)

### 第 2 轮: 训练策略优化

#### 2.1 NWD Loss（归一化 Wasserstein 距离）
- **原理**: 将 bbox 建模为 2D 高斯分布，用 Wasserstein 距离衡量相似度。检测实验证明对小目标有效
- **实现**: 通过 monkey-patch 混入 CIoU loss: `loss = (1-0.05)×CIoU + 0.05×NWD`
- **Baseline+NWD**: Mask mAP50=0.842 (+0.006), mAP50-95=0.509 (-0.020)
- **P3LCRefine+NWD**: Mask mAP50=0.847 (+0.011), mAP50-95=0.508 (-0.021)
- **分析**: NWD 持续提升 mAP50（检测更多气泡）但损害 mAP50-95（定位精度下降）。存在 recall-precision tradeoff

#### 2.2 更高分辨率 (768px)
- **原理**: 检测实验证明 768 > 640，因为气泡是小目标
- **配置**: imgsz=768, batch=2（显存限制）
- **结果**: Mask mAP50=0.838, mAP50-95=0.521 (-0.008)
- **分析**: 768px 下 batch=2 太小，BatchNorm 统计不稳定，可能抵消了分辨率收益

### 第 3 轮: gamma 调优（COCO 预训练保护）

#### 3.1 降低 P3LCRefine 的 gamma_init
- **原理**: 检测实验中的 COCO 预训练策略是先在 COCO 上训练含 custom module 的完整架构 30 epoch，再迁移到气泡。我们跳过了 COCO 预训练步骤，直接训练。降低 gamma_init 可让模块接近恒等映射启动，保护 COCO 特征
- **gamma=0.01**: Mask mAP50=0.835, mAP50-95=0.513
- **gamma=0.1**: Mask mAP50=0.830, mAP50-95=0.512
- **分析**: 降低 gamma 无帮助，原始 gamma=1.0 最好。这暗示 P3LCRefine 的收益来自"立即"的对比度增强，而非渐进适应

### 第 4 轮: 检测 Pipeline 方法迁移

深度分析了 72 个检测实验后，发现检测最佳配置与分割完全不同。

#### 4.1 SGD + cos_lr + 零增强
- **原理**: 检测 top 实验全部使用 SGD (momentum=0.937) + 余弦退火 + 零增强 (mosaic=0, mixup=0, hsv=0)。我们原来用 AdamW + 中等增强
- **结果**: Mask mAP50=0.829, mAP50-95=0.528 (几乎追平 baseline 0.529)
- **关键**: 收敛速度提升 4.6 倍（epoch 10 vs 46）
- **分析**: SGD 对气泡数据泛化更好，零增强避免破坏气泡形态

#### 4.2 检测 Backbone 迁移
- **原理**: 使用检测最佳 checkpoint（PV4C1: P3LCRefine + NWD + 强增强 + SGD + 768px, 在气泡检测上已收敛）初始化分割 backbone。backbone/neck 架构与分割模型完全一致
- **结果**: Mask mAP50=0.834, mAP50-95=0.499
- **分析**: 检测特征与分割不兼容。可能原因: 检测模型含 P3LCRefine 模块（分割模型无），权重匹配不完整；768px 特征迁移到 640px 有分辨率差距

---

## 五、核心瓶颈分析

### 瓶颈 1: mAP50-95 难以突破 0.53
所有 12 个实验的 mAP50-95 都落在 0.50-0.53 范围内。这个指标衡量的是严格 IoU 阈值 (0.50→0.95) 下的平均精度，反映**掩码定位精度**。架构修改总是降低这个指标。

**可能原因**:
- 气泡边界本身模糊（光学成像限制），无论模型如何改进都有精度上限
- SAM3 生成的 polygonal mask 存在系统误差（polygon 简化到 ≤50 顶点）
- 瓦片化处理 (stride=480) 在边界处切割气泡，引入标注噪声

### 瓶颈 2: 数据量小
仅 170 张原始图片（712 张瓦片）。检测实验也是 170 张但达到更好的指标。分割任务更难，可能需要更多数据。

### 瓶颈 3: 自定义模块在分割上无效
16 个自定义模块（SSBRefine、GLRB、LCRefine、各种注意力门控等）在检测任务上有帮助（+0.01-0.02 mAP50），但在分割上完全无效甚至有害。

**可能原因**:
- 这些模块设计用于增强检测特征（分类+回归），而不是掩码预测所需的细粒度空间特征
- 模块放在 P3 层，直接输入 Segment head 的 proto mask 生成器，可能干扰了 mask 特征的表达能力

---

## 六、尚未验证的假设

### 假设 A: 数据端改进
- **[ ] 增加标注数据**: 最直接的方向。170 张图对分割任务可能不够
- **[ ] 改进 SAM3 标注质量**: 当前 polygon 简化到 ≤50 顶点，可能丢失边界细节。尝试更多顶点或使用 SAM3 原始 mask
- **[ ] 减少瓦片化噪声**: 当前 stride=480 在边界切分气泡。尝试不同的瓦片化策略或不瓦片化（但需解决大图显存问题）
- **[ ] 修复瓦片重叠区域的重复标注**: build_dataset.py 已移除重复标注，但可能不完美

### 假设 B: 训练端改进
- **[ ] Dice Loss / Boundary Loss**: 论文表明 Dice Loss 直接优化 mask IoU，可能比 BCE 更适合气泡分割
- **[ ] 两阶段训练**: 先在 COCO 分割上训练 custom module（但需要大量 GPU 时间），再迁移到气泡
- **[ ] 冻结 backbone 逐步解冻**: "Fine-Tuning Without Forgetting" (2025) 论文证明渐进解冻可提升 +10% mAP50
- **[ ] EMA (Exponential Moving Average)**: 许多分割论文使用 EMA 权重获得更好泛化
- **[ ] 更长的 warmup + 更小的 peak LR**: 当前 warmup=3 epoch, lr0=0.001。检测实验的最佳配置类似但数据更少

### 假设 C: 架构端改进
- **[ ] P2 检测头 (160×160 特征图)**: 论文一致表明对 small object segmentation 有效（+3-5% mAP）。当前最低只有 P3 (80×80)
- **[ ] CARAFE / 内容感知上采样**: 替换 FPN 中的 nearest-neighbor 上采样，更好地保留空间细节
- **[ ] ProtoNet 改进**: 修改 Segment head 的 proto mask 数量和隐藏通道。当前 32 proto/256 hidden，调整可能有用
- **[ ] SPD-Conv (space-to-depth convolution)**: 替代 strided convolution 减少下采样信息损失
- **[ ] 双路径架构**: YOLOE (2025) 或 Dual-Path YOLO11 (2025) 增加独立的分割分支

### 假设 D: 推理端改进
- **[ ] 测试时增强 (TTA)**: 多尺度 + 翻转融合推理，通常可提升 1-2 个点
- **[ ] SAHI 推理**: 对大图进行重叠瓦片推理 + NMS 融合，对小目标特别有效
- **[ ] Mask 后处理**: CRF 或形态学操作优化边界

---

## 七、环境中可用的资源

### 自定义模块库 (`ultralytics_custom/`)
- 16 个自定义 PyTorch 模块: SSBRefine, GLRB, LCRefine/P3LCRefine, MSLRefine, P3MLCRefine, P3CAGate, P3SAGate, ECAGate, CoordGate, SimAMGate, WeightedConcat, ChannelWeightedConcat, LayerNorm2d, MDTA, GDFN
- NWD loss 实现 (Normalized Wasserstein Distance) + WIoU v3
- Weight transfer 机制（但目前只支持 Detection head，不支持 Segment head）

### 检测实验结果
- 72 个检测实验的完整记录: `runs/summary/paper_v4_all_experiment_summary.csv`
- 最佳检测模型: PV4C1_P3LC_COCO_NWD_W005_MOSAIC10_HSV_MIXUP_E30_seed48 (map50=0.838 on main test)

### GPU
- 2x Tesla V100 16GB VRAM
- 训练可用 batch=4@640px 或 batch=2@768px
- 无法进行完整 COCO 分割预训练（需数天）

---

## 八、关键文件索引

| 文件 | 用途 |
|------|------|
| `segmentation/EXPERIMENT_SUMMARY.md` | 12 个实验详细结果 |
| `segmentation/generate_masks.py` | SAM3 推理脚本 |
| `segmentation/build_dataset.py` | 数据集构建脚本 |
| `segmentation/scripts/train_seg.py` | 训练入口 |
| `segmentation/configs/models/` | 模型 YAML 配置 (8 个) |
| `segmentation/configs/train/` | 训练 YAML 配置 (5 个) |
| `ultralytics_custom/bubble_modules.py` | 16 个自定义模块 |
| `ultralytics_custom/bubble_loss.py` | NWD + WIoU 损失 |
| `ultralytics_custom/weight_transfer.py` | 权重迁移 |
| `runs/summary/paper_v4_all_experiment_summary.csv` | 检测实验完整结果 |
