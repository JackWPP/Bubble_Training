# Bubble-YOLO11s 训练与模型改进实施规格书

> 目的：本文档用于交给 Codex 执行后续编程工作。核心任务是基于现有综合气泡检测数据集，训练 YOLO11s baseline，并逐步实现参考 HSMD-YOLO 思路的可落地模型改进，包括 SSB、GLRB 和 NWD loss。本文档不要求 100% 复现 HSMD-YOLO，而是要求实现“功能等价、工程可验证、训练稳定”的 Bubble-YOLO11s。

---

## 0. 项目背景与基本判断

当前项目是单类别气泡检测任务，类别统一为 `bubble`，YOLO 类别编号为 `0`。现有数据集已经由 7 个 COCO 子数据集合并，并转换为 YOLO 格式，输出目录为：

```text
G:\Bubble_Train\yolo_dataset_integrated
```

训练配置文件为：

```text
G:\Bubble_Train\yolo_dataset_integrated\bubble.yaml
```

当前综合数据集已经完成高分辨率图像切片：任一边大于 `640` 的图像使用 `640x640` 滑动窗口切片，stride 为 `480`；小图保持比例缩放并 padding 到 `640x640`。训练集做离线增强，验证集和测试集保持未增强。当前输出统计为：

```text
train: 5715 images, 176145 boxes
val:     84 images,   1908 boxes
test:    23 images,    515 boxes
```

构建后 bbox 宽高中位数约为：

```text
bbox width p50 ≈ 41.6 px
bbox height p50 ≈ 40.55 px
```

这说明现有 `640x640` 切片策略已经把原始高分辨率图中的小气泡放大到相对可学习的尺度。后续核心不再是继续重构数据，而是：

1. 训练一个稳定、可复现的 YOLO11s baseline；
2. 在 baseline 上逐步加入少量结构改进；
3. 用严格消融证明每个改动的作用；
4. 避免一次性堆模块导致结果不可解释。

---

## 1. 总体技术路线

### 1.1 主 baseline

主 baseline 使用：

```text
YOLO11s
```

不再使用 nano 模型。原因：本课题目标是“气泡检测模型改进”，不是边缘端轻量化部署。nano 模型容量偏小，不适合作为最终主线。Small 尺寸模型更适合密集小目标、弱边界和复杂背景检测。

### 1.2 改进模型名称

建议最终模型命名为：

```text
Bubble-YOLO11s
```

建议论文表述：

```text
以 YOLO11s 为基础检测器，针对气泡图像中小目标密集、弱边界、重叠粘连和高分辨率输入下特征易损失的问题，引入尺度变换增强模块、全局-局部特征精炼模块以及小目标友好的边界框回归损失，构建改进型气泡检测模型 Bubble-YOLO11s。
```

### 1.3 改进模块优先级

| 优先级 | 模块 | 是否实现 | 工程风险 | 说明 |
|---:|---|---|---:|---|
| 1 | SSB / Scale Switch 思想模块 | 必做 | 低~中 | 先做简化稳定版，解决采样后边界弱化和伪纹理问题 |
| 2 | GLRB / Global-Local Refine Block | 必做 | 中 | 参考 Restormer 的 MDTA + GDFN，作为主要注意力机制 |
| 3 | NWD loss | 必做或强烈建议 | 中 | 解决小框 IoU 敏感、定位不稳定问题 |
| 4 | BEMAF | 暂不做 | 高 | 多尺度 EMA attention 融合复杂，收益相对有限，先放弃 |
| 5 | P2 detection head | 可选 | 中~高 | 仅当小目标 recall 仍明显不足时再考虑 |

不要一上来实现 BEMAF，也不要同时实现所有模块。Codex 应按阶段提交，每个阶段必须能单独训练、单独验证。

---

## 2. 代码仓库约定

现有 README 中约定：

```text
Dataset/                 本地原始 COCO 数据集来源，不进 git
yolo_dataset_integrated/ 生成后的综合 YOLO 数据集，不进 git
runs/                   Ultralytics 训练和验证输出，不进 git
DATASET_BUILD_REPORT.md 数据集报告
```

后续建议新增：

```text
configs/
  models/
    bubble_yolo11s.yaml
    bubble_yolo11s_ssb.yaml
    bubble_yolo11s_glrb.yaml
    bubble_yolo11s_ssb_glrb.yaml
  train/
    baseline_yolo11s.yaml
    ssb_yolo11s.yaml
    glrb_yolo11s.yaml
    ssb_glrb_yolo11s.yaml
    final_bubble_yolo11s.yaml

scripts/
  train_yolo11s_baseline.py
  train_experiment.py
  val_model.py
  predict_compare.py

tools/
  make_debug_subset.py
  check_model_forward.py
  collect_results.py
  visualize_predictions.py
  compare_predictions.py

ultralytics_custom/
  bubble_modules.py
  bubble_loss.py
```

如果当前项目直接依赖 pip 安装的 `ultralytics`，不要直接修改 site-packages。建议让 Codex 采用以下二选一方案：

方案 A，推荐：clone Ultralytics 到本地并 editable install。

```powershell
cd G:\Bubble_Train
git clone https://github.com/ultralytics/ultralytics.git ultralytics_src
cd ultralytics_src
pip install -e .
```

然后在 `ultralytics_src/ultralytics/nn/modules/` 和 `ultralytics_src/ultralytics/nn/tasks.py` 中注册自定义模块。

方案 B：在当前仓库中复制必要模块并通过 monkey patch 注册。该方案不推荐，维护成本高。

---

## 3. 外部资源与可参考代码

### 3.1 Ultralytics YOLO11

用途：baseline、训练框架、自定义模型 YAML、自定义模块注册。

参考：

```text
https://docs.ultralytics.com/models/yolo11/
https://docs.ultralytics.com/guides/model-yaml-config/
https://github.com/ultralytics/ultralytics
```

需要 Codex 重点查看：

```text
ultralytics/cfg/models/11/yolo11.yaml
ultralytics/nn/tasks.py
ultralytics/nn/modules/block.py
ultralytics/nn/modules/conv.py
ultralytics/utils/loss.py
```

注意：Ultralytics 使用 AGPL-3.0 / Enterprise 双许可。学术研究可用；商业闭源部署要注意许可。

### 3.2 HSMD-YOLO 论文

用途：问题建模和模块设计思想，不假设有官方代码。

核心可参考点：

```text
SSB: 放在 up/down sampling 相关位置，抑制采样伪高频、背景噪声和边界破坏。
GLRB: 使用 MDTA + GDFN，实现全局关系建模和局部细节精炼。
BEMAF: 多尺度 EMA attention 融合，暂不实现。
NWD loss: 用于小目标框回归稳定，论文中 ratio=0.4 效果较均衡。
```

### 3.3 Restormer

用途：GLRB 中 MDTA 与 GDFN 的成熟实现参考。

参考：

```text
https://github.com/swz30/Restormer
https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
```

Codex 不应直接照搬完整 Restormer 网络，只需要移植以下模块思想：

```text
LayerNorm2d
MDTA: Multi-DConv Head Transposed Self-Attention
GDFN: Gated-DConv Feed-Forward Network
```

### 3.4 NWD / Normalized Gaussian Wasserstein Distance

用途：小目标检测中的框回归损失参考。

参考：

```text
https://arxiv.org/abs/2110.13389
https://github.com/jwwangchn/NWD
```

建议只实现 NWD loss，不要一开始实现 NWD-NMS 或 NWD-based assignment。

### 3.5 LGA-YOLO

用途：训练配置、实验组织、轻量化参考，不作为主线。

可参考：

```text
训练 200 epochs
imgsz=640
SGD lr=0.01 momentum=0.937
指标：Precision, Recall, mAP@50, mAP@50-95, Params, FLOPs
```

不要参考它的主创新方向作为本课题主线，因为 LGA-YOLO 重点是轻量化，而本课题重点是提升气泡检测上限。

---

## 4. 训练阶段规划

### 4.1 实验编号

必须按以下顺序执行：

| 编号 | 模型 | 目的 |
|---|---|---|
| E0 | YOLO11s baseline | 建立主基线 |
| E1 | YOLO11s + SSB-Safe | 验证尺度变换/抗混叠思想 |
| E2 | YOLO11s + GLRB-P3 | 验证注意力式全局-局部特征精炼 |
| E3 | YOLO11s + SSB-Safe + GLRB-P3 | 验证组合效果 |
| E4 | YOLO11s + SSB-Safe + GLRB-P3/P4/P5 | 验证多尺度插入效果 |
| E5 | YOLO11s + SSB + GLRB + NWD loss | 最终模型 |

如果时间不足，压缩为：

| 编号 | 模型 |
|---|---|
| E0 | YOLO11s baseline |
| E1 | YOLO11s + SSB-Safe |
| E2 | YOLO11s + SSB-Safe + GLRB-P3 |
| E3 | YOLO11s + SSB-Safe + GLRB-P3 + NWD loss |

### 4.2 训练超参数建议

统一默认配置：

```yaml
model: yolo11s.pt 或 configs/models/bubble_yolo11s*.yaml
data: G:\Bubble_Train\yolo_dataset_integrated\bubble.yaml
imgsz: 640
epochs: 200
batch: 8 或 16，根据显存自动调整
workers: 4 或 8
device: 0
seed: 42
patience: 50
pretrained: true
optimizer: SGD 或 auto
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
cos_lr: true
close_mosaic: 20
```

注意：当前数据集已经有离线增强，因此在线增强不要过强。建议：

```yaml
mosaic: 0.3
mixup: 0.0
copy_paste: 0.0
hsv_h: 0.005
hsv_s: 0.2
hsv_v: 0.2
degrees: 0.0
translate: 0.05
scale: 0.2
fliplr: 0.5
flipud: 0.0
```

说明：数据集报告中已经包含水平翻转、垂直翻转、90/180/270 度旋转、亮度/对比度/HSV、轻噪声和 copy-paste 离线增强。训练时不应再叠加强旋转和 copy-paste，否则样本可能偏离真实气泡成像。

---

## 5. Baseline 实施细则

### 5.1 训练脚本

新增：

```text
scripts/train_yolo11s_baseline.py
```

功能：

1. 读取环境变量或默认路径；
2. 使用 `YOLO('yolo11s.pt')`；
3. 使用当前综合数据集训练；
4. 固定随机种子；
5. 输出实验名 `E0_yolo11s_baseline`；
6. 训练结束后自动执行 val；
7. 保存结果摘要到 `runs/summary/E0_yolo11s_baseline.json`。

伪代码：

```python
import os
from ultralytics import YOLO

DATA = os.getenv('BUBBLE_DATA_CONFIG', r'G:\Bubble_Train\yolo_dataset_integrated\bubble.yaml')
WEIGHTS = os.getenv('BUBBLE_PRETRAINED_WEIGHTS', r'G:\Bubble_Train\weights\yolo11s.pt')
DEVICE = os.getenv('BUBBLE_DEVICE', '0')

model = YOLO(WEIGHTS)
model.train(
    data=DATA,
    imgsz=640,
    epochs=200,
    batch=8,
    device=DEVICE,
    seed=42,
    patience=50,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    mosaic=0.3,
    close_mosaic=20,
    mixup=0.0,
    copy_paste=0.0,
    project='runs/bubble',
    name='E0_yolo11s_baseline',
)
model.val(data=DATA, imgsz=640, device=DEVICE, split='test')
```

### 5.2 Baseline 验收标准

Baseline 必须满足：

```text
1. 训练不报错。
2. loss 正常下降。
3. val/test 输出 Precision, Recall, mAP@50, mAP@50-95。
4. 能生成预测可视化图。
5. 能用 model.info() 输出 Params 和 FLOPs。
6. 保存 best.pt 和 last.pt。
```

如果 baseline 训练不稳定，不允许进入模块开发阶段。

---

## 6. 自定义模块接入 Ultralytics 的通用规范

Codex 必须先检查当前 Ultralytics 版本的 `parse_model()` 逻辑。不同版本中自定义模块注册方式可能不同。

通常需要修改：

```text
ultralytics/nn/modules/bubble_modules.py      # 新增自定义模块
ultralytics/nn/modules/__init__.py            # 导出模块
ultralytics/nn/tasks.py                       # 让 parse_model 能解析自定义模块
configs/models/bubble_yolo11s_*.yaml          # 使用自定义模块
```

### 6.1 注册原则

如果模块需要输入通道 `c1` 和输出通道 `c2`，则需要在 `parse_model()` 中加入类似逻辑：

```python
if m in {SSBRefine, SSBDown, SSBUp, GLRB}:
    c1 = ch[f]
    # 对于 shape-preserving 模块，c2 = c1
    # 对于改变通道模块，c2 从 args[0] 读取
```

Codex 不得盲目写 YAML。必须先打印每层通道，确认自定义模块输入输出 shape 正确。

### 6.2 必须新增检查脚本

新增：

```text
tools/check_model_forward.py
```

功能：

```python
from ultralytics import YOLO
import torch

model = YOLO('configs/models/bubble_yolo11s_ssb.yaml')
model.info(verbose=True)
_ = model.model(torch.randn(1, 3, 640, 640))
print('forward ok')
```

---

## 7. SSB 模块实施方案

### 7.1 不建议直接硬复刻 HSMD-SSB

HSMD 论文中的 SSB 描述清楚，但未必有官方代码。直接复刻容易出现以下问题：

```text
1. 与 Ultralytics YOLO11 的通道解析不兼容；
2. 替换下采样后改变 feature map 尺寸或通道；
3. 训练初期不稳定；
4. 出现 shape 对不上但原因难查。
```

因此第一版实现采用安全等价策略：

```text
SSB-Safe = shape-preserving Scale Refinement Block
```

它不直接替换所有上采样/下采样，而是插入在 neck 融合后的特征层，用可学习局部卷积增强采样后的边界细节。

### 7.2 SSB-Safe 目标

模块目标：

```text
输入:  [B, C, H, W]
输出:  [B, C, H, W]
作用:  抑制采样后伪纹理，增强局部边界和薄壁气泡轮廓。
```

### 7.3 SSB-Safe 结构

建议实现：

```text
x
├── main: Conv 3x3 -> BN -> SiLU -> DWConv 3x3 -> BN -> SiLU -> Conv 1x1
└── skip: identity
out = x + gamma * main(x)
```

其中 `gamma` 是可学习标量或初始化为 0.1 的参数，保证模块初期不会破坏 baseline。

### 7.4 参考代码骨架

```python
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv

class SSBRefine(nn.Module):
    """Shape-preserving Scale Switch inspired refinement block.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, c1, c2=None, expansion=1.0, shortcut=True):
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = int(c2 * expansion)
        self.proj_in = Conv(c1, hidden, k=3, s=1)
        self.dw = DWConv(hidden, hidden, k=3, s=1)
        self.proj_out = Conv(hidden, c2, k=1, s=1)
        self.shortcut = shortcut and c1 == c2
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.proj_out(self.dw(self.proj_in(x)))
        if self.shortcut:
            return x + self.gamma * y
        return y
```

注意：`DWConv` 的参数签名需要 Codex 根据当前 Ultralytics 版本核对。若不兼容，用 `nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)` 自行实现。

### 7.5 SSB-Safe 插入位置

优先插在 YOLO11s neck 中 P3/P4/P5 进入 Detect 前的 C3k2 后面。

抽象结构：

```text
原结构：
C3k2 -> Detect

改为：
C3k2 -> SSBRefine -> Detect
```

第一阶段只插 P3：

```text
C3k2(P3) -> SSBRefine -> Detect(P3)
```

如果训练稳定，再插 P4/P5。

### 7.6 SSBDown / SSBUp 可选增强

只有当 SSB-Safe 训练稳定后，才考虑直接实现：

```text
SSBDown: 替代 stride=2 Conv 或部分下采样模块
SSBUp:   替代 Upsample + Conv 或作为上采样后的 refinement
```

不建议作为第一版。

---

## 8. GLRB 模块实施方案

### 8.1 GLRB 目标

GLRB 是本项目最重要的注意力机制模块。目标：

```text
输入:  [B, C, H, W]
输出:  [B, C, H, W]
作用:  同时建模全局上下文和局部边界细节，提升密集、重叠、弱边界气泡检测。
```

### 8.2 GLRB 结构

参考 Restormer 的 TransformerBlock：

```text
x = x + MDTA(LayerNorm2d(x))
x = x + GDFN(LayerNorm2d(x))
```

其中：

```text
MDTA: Multi-DConv Head Transposed Self-Attention
GDFN: Gated-DConv Feed-Forward Network
```

### 8.3 MDTA 设计要点

MDTA 不是普通空间 self-attention。它不计算 `[HW, HW]` 的注意力矩阵，而是在通道维度计算注意力，降低高分辨率下的计算压力。

输入输出：

```text
Input:  [B, C, H, W]
Output: [B, C, H, W]
```

核心流程：

```text
1. 1x1 Conv 生成 QKV，通道 C -> 3C
2. 3x3 depthwise Conv 做局部空间混合
3. q,k,v reshape 为 [B, heads, C_per_head, H*W]
4. q,k 在最后一维 normalize
5. attention = softmax(q @ k^T * temperature)
6. out = attention @ v
7. reshape 回 [B, C, H, W]
8. 1x1 Conv 输出
```

### 8.4 GDFN 设计要点

GDFN 用于局部细节和非线性特征选择。

核心流程：

```text
1. 1x1 Conv: C -> hidden*2
2. 3x3 depthwise Conv
3. split 成 x1, x2
4. GELU(x1) * x2
5. 1x1 Conv: hidden -> C
```

### 8.5 参考代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by heads {num_heads}'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class GLRB(nn.Module):
    def __init__(self, c1, c2=None, num_heads=4, expansion_factor=2.0, bias=False):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, 'First implementation should be shape-preserving only.'
        assert c1 % num_heads == 0, 'Channels must be divisible by num_heads.'
        self.norm1 = LayerNorm2d(c1)
        self.attn = MDTA(c1, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(c1)
        self.ffn = GDFN(c1, expansion_factor=expansion_factor, bias=bias)
        self.gamma1 = nn.Parameter(torch.tensor(0.1))
        self.gamma2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x
```

### 8.6 GLRB 插入位置

第一版只插入 P3：

```text
P3 C3k2 -> GLRB -> Detect
```

原因：P3 是 YOLO 检测小目标的主要高分辨率层，收益最可能出现，同时计算量可控。

第二版再插 P4/P5：

```text
P3 C3k2 -> GLRB -> Detect
P4 C3k2 -> GLRB -> Detect
P5 C3k2 -> GLRB -> Detect
```

### 8.7 heads 设置建议

Codex 需要根据实际通道数设置 `num_heads`：

```text
C=128 -> heads=4
C=256 -> heads=4 或 8
C=512 -> heads=8
```

若通道不能整除 heads，自动降级：

```python
def choose_heads(c):
    for h in [8, 4, 2, 1]:
        if c % h == 0:
            return h
    return 1
```

---

## 9. NWD loss 实施方案

### 9.1 目标

YOLO 默认 box loss 基于 IoU 类指标，小目标框稍微偏移就可能导致 IoU 大幅下降。NWD 使用高斯 Wasserstein 距离度量预测框和真实框的距离，对小目标更平滑。

### 9.2 实现原则

不要完全替换原 box loss。采用融合：

```text
box_loss = (1 - nwd_weight) * original_box_loss + nwd_weight * nwd_loss
```

建议第一版：

```text
nwd_weight = 0.4
nwd_constant = 12.8  # 可作为默认值，后续可调
```

### 9.3 NWD 计算公式工程版

假设 box 格式为 `xywh`：

```python
def bbox_nwd_xywh(pred_xywh, target_xywh, C=12.8, eps=1e-7):
    # pred_xywh, target_xywh: [..., 4], xywh in same scale
    pxy = pred_xywh[..., 0:2]
    txy = target_xywh[..., 0:2]
    pwh = pred_xywh[..., 2:4].clamp(min=eps)
    twh = target_xywh[..., 2:4].clamp(min=eps)

    center_dist = ((pxy - txy) ** 2).sum(dim=-1)
    size_dist = ((pwh - twh) ** 2).sum(dim=-1) / 4.0
    wasserstein = torch.sqrt(center_dist + size_dist + eps)
    nwd = torch.exp(-wasserstein / C)
    return nwd
```

loss：

```python
nwd_loss = 1.0 - bbox_nwd_xywh(pred_xywh, target_xywh, C=nwd_constant)
```

注意：

```text
1. pred 和 target 必须在同一尺度下，例如都是像素尺度或都是同一归一化尺度。
2. 如果 Ultralytics loss 内部使用 xyxy，需要先转换为 xywh。
3. 接入前必须打印几个样本的 pred/target 范围，确认尺度一致。
4. 如果 loss 出现 NaN，优先检查 w/h 是否为 0 或尺度是否错。
```

### 9.4 接入位置

Codex 应检查：

```text
ultralytics/utils/loss.py
```

通常 box loss 发生在 `BboxLoss` 或类似类中。实现步骤：

1. 找到原始 IoU loss 的计算位置；
2. 保留原 loss；
3. 新增 NWD 计算；
4. 增加超参数 `nwd_weight` 和 `nwd_constant`；
5. 输出训练日志中记录当前是否启用 NWD。

### 9.5 NWD 实验

建议做两个实验：

```text
E5a: nwd_weight=0.4
E5b: nwd_weight=0.5
```

如果时间不足，只做 `0.4`。

---

## 10. 模型 YAML 改造规范

### 10.1 基础文件

从官方 YOLO11s yaml 复制：

```text
ultralytics/cfg/models/11/yolo11.yaml
```

生成：

```text
configs/models/bubble_yolo11s.yaml
```

不要修改官方原始 yaml。

### 10.2 SSB YAML

生成：

```text
configs/models/bubble_yolo11s_ssb.yaml
```

要求：

```text
1. 先只插入 SSBRefine，不直接替换 Upsample 或 stride Conv。
2. 插入位置优先为 neck 中 P3/P4/P5 的 C3k2 后。
3. 第一版只插 P3，第二版再插 P4/P5。
```

### 10.3 GLRB YAML

生成：

```text
configs/models/bubble_yolo11s_glrb_p3.yaml
configs/models/bubble_yolo11s_glrb_all.yaml
```

要求：

```text
1. GLRB 初版必须 shape-preserving。
2. 插入后 Detect 的输入层索引必须重新检查。
3. 每改一次 YAML，必须跑 tools/check_model_forward.py。
```

### 10.4 最终 YAML

生成：

```text
configs/models/bubble_yolo11s_final.yaml
```

结构为：

```text
YOLO11s + SSBRefine + GLRB + Detect
```

NWD 不写在模型 YAML 中，通过训练脚本或 loss 配置启用。

---

## 11. 调试与验证流程

每一个模块加入后，必须按以下顺序检查。

### 11.1 Shape 单元测试

新增：

```text
tests/test_bubble_modules.py
```

测试内容：

```python
def test_ssb_refine_shape():
    x = torch.randn(2, 128, 80, 80)
    m = SSBRefine(128)
    y = m(x)
    assert y.shape == x.shape


def test_glrb_shape():
    x = torch.randn(2, 128, 80, 80)
    m = GLRB(128, num_heads=4)
    y = m(x)
    assert y.shape == x.shape


def test_glrb_no_nan():
    x = torch.randn(2, 128, 80, 80)
    m = GLRB(128, num_heads=4)
    y = m(x)
    assert torch.isfinite(y).all()
```

### 11.2 模型 forward 测试

每个 yaml 都必须通过：

```powershell
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_ssb.yaml
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_glrb_p3.yaml
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_final.yaml
```

通过标准：

```text
1. model.info() 能输出。
2. torch.randn(1, 3, 640, 640) 能 forward。
3. 没有 shape mismatch。
4. 没有 NaN。
```

### 11.3 小样本过拟合测试

新增：

```text
tools/make_debug_subset.py
```

生成一个 20~50 张图的小数据集：

```text
yolo_dataset_debug/
  train/images
  train/labels
  val/images
  val/labels
  bubble_debug.yaml
```

每个新模型必须先在 debug 数据集训练 30~50 epoch。

通过标准：

```text
1. train loss 明显下降。
2. 预测框能逐渐贴近目标。
3. 没有 loss NaN。
4. 没有显存异常。
```

### 11.4 全量短训测试

每个新模型先跑 20 epoch：

```text
E1_ssb_smoke_20epoch
E2_glrb_smoke_20epoch
```

通过后再跑 200 epoch 正式训练。

### 11.5 可视化检查

必须固定同一批测试图，可视化对比：

```text
small_dense/       小气泡密集
overlap/           粘连重叠
weak_boundary/     弱边界低对比
highlight_noise/   高光/反光/复杂背景
```

新增脚本：

```text
tools/compare_predictions.py
```

输入：

```powershell
python .\tools\compare_predictions.py ^
  --models runs\bubble\E0_yolo11s_baseline\weights\best.pt runs\bubble\E5_final\weights\best.pt ^
  --source G:\Bubble_Train\selected_test_images ^
  --out runs\bubble\compare_E0_E5
```

输出：

```text
原图
baseline prediction
improved prediction
并排图
```

---

## 12. 结果记录规范

每个实验必须记录：

```text
experiment_id
model_yaml
pretrained_weight
data_config
imgsz
epochs
batch
optimizer
lr0
seed
augmentation config
Params
FLOPs
Precision
Recall
mAP@50
mAP@50-95
best epoch
training time
inference time/FPS
notes
```

建议生成：

```text
runs/bubble/experiment_summary.csv
```

列：

```csv
exp_id,model,modules,nwd_weight,params,flops,precision,recall,map50,map5095,best_epoch,train_time,notes
```

最终论文表格建议：

| 实验 | 模型 | SSB | GLRB | NWD | Params | FLOPs | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| E0 | YOLO11s | ✗ | ✗ | ✗ | | | | | | |
| E1 | YOLO11s+SSB | ✓ | ✗ | ✗ | | | | | | |
| E2 | YOLO11s+GLRB | ✗ | ✓ | ✗ | | | | | | |
| E3 | YOLO11s+SSB+GLRB | ✓ | ✓ | ✗ | | | | | | |
| E5 | Bubble-YOLO11s | ✓ | ✓ | ✓ | | | | | | |

---

## 13. 训练失败时的排查顺序

### 13.1 训练 loss 不下降

按顺序检查：

```text
1. baseline 是否正常；如果 baseline 不正常，先回到数据和训练脚本。
2. 自定义模块输出是否全 0 或数值过大。
3. gamma 是否初始化太大；建议 0.1 或 0.0。
4. 学习率是否过高；可降低 lr0 到 0.005。
5. 在线增强是否过强；降低 mosaic/scale/translate。
```

### 13.2 出现 NaN

按顺序检查：

```text
1. GLRB attention 中 softmax 前是否数值过大。
2. LayerNorm2d eps 是否太小。
3. NWD 中 w/h 是否出现 0。
4. sqrt 是否加 eps。
5. mixed precision 是否导致不稳定；可临时 amp=False。
```

### 13.3 显存爆炸

按顺序降低：

```text
1. batch size 从 16 降到 8，再降到 4。
2. GLRB 只保留 P3。
3. GLRB expansion_factor 从 2.0 降到 1.5。
4. num_heads 降低。
5. 暂时关闭 GLRB，只测试 SSB。
```

### 13.4 mAP 下降

不要立刻否定模块，先看：

```text
1. Recall 是否升高但 Precision 降低。
2. mAP@50 是否升高但 mAP@50-95 下降。
3. 小气泡密集区域是否改善。
4. 高光区域误检是否增加。
5. 是否需要调 conf / iou / NMS 阈值。
```

如果 E1/E2 单模块下降，但 E3 组合提升，也可以保留组合；但论文中必须如实解释。

---

## 14. 推理与后处理策略

主训练仍使用 `640x640` patch 数据。最终在原始高分辨率图上推理时，建议保留滑窗/切片推理策略。

新增或完善：

```text
scripts/predict_tiled.py
```

功能：

```text
1. 输入原始 1920x1080 或其他高分辨率图；
2. 使用 640x640 window, stride=480 切片；
3. 每个 tile 独立推理；
4. 将 tile 坐标映射回原图；
5. 对全图检测框做 NMS 或 Soft-NMS；
6. 输出原图级检测结果和可视化图。
```

优先实现普通 NMS，Soft-NMS 作为可选。

建议评估不同阈值：

```text
conf: 0.15, 0.25, 0.35
iou:  0.5, 0.6, 0.7
```

气泡密集重叠场景下，过低 NMS IoU 可能误删相邻气泡。

---

## 15. 不建议当前实现的内容

当前阶段不要做：

```text
1. 完整复现 HSMD-YOLO 全部模块。
2. BEMAF 多尺度 EMA attention 融合。
3. 完整 DETR / RT-DETR 主线替换。
4. YOLO11s + 大量随机注意力模块堆叠。
5. 同时修改模型结构、loss、数据增强和 NMS。
6. 没有 baseline 就直接训练最终模型。
7. 没有单元测试就全量训练。
```

这些都会导致结果不可解释或工程风险过高。

---

## 16. Codex 分阶段任务拆解

### Task 1：整理 baseline 训练入口

目标：创建 YOLO11s baseline 训练脚本。

产出：

```text
scripts/train_yolo11s_baseline.py
scripts/val_model.py
runs/bubble/E0_yolo11s_baseline/
```

验收：

```text
1. 能正常训练。
2. 能正常 val/test。
3. 保存 summary json/csv。
```

### Task 2：建立自定义模块接入框架

目标：让 Ultralytics 能识别自定义模块。

产出：

```text
ultralytics/nn/modules/bubble_modules.py
ultralytics/nn/modules/__init__.py 修改
tasks.py parse_model 修改
configs/models/bubble_yolo11s.yaml
```

验收：

```text
python tools/check_model_forward.py --model configs/models/bubble_yolo11s.yaml
```

### Task 3：实现 SSBRefine

目标：实现 shape-preserving SSB-inspired refinement block。

产出：

```text
SSBRefine 类
configs/models/bubble_yolo11s_ssb_p3.yaml
configs/models/bubble_yolo11s_ssb_all.yaml
```

验收：

```text
pytest tests/test_bubble_modules.py
python tools/check_model_forward.py --model configs/models/bubble_yolo11s_ssb_p3.yaml
debug subset 训练 50 epoch 正常
```

### Task 4：实现 GLRB

目标：实现 Restormer-style MDTA + GDFN。

产出：

```text
LayerNorm2d
MDTA
GDFN
GLRB
configs/models/bubble_yolo11s_glrb_p3.yaml
configs/models/bubble_yolo11s_ssb_glrb_p3.yaml
```

验收：

```text
pytest tests/test_bubble_modules.py
forward 测试正常
smoke train 20 epoch 正常
```

### Task 5：实现 NWD loss

目标：在原 box loss 上融合 NWD loss。

产出：

```text
ultralytics_custom/bubble_loss.py 或直接修改 ultralytics/utils/loss.py
train_experiment.py 支持 --use-nwd --nwd-weight --nwd-constant
```

验收：

```text
1. use_nwd=False 时结果与原 loss 一致。
2. use_nwd=True 时 loss 正常下降。
3. 无 NaN。
```

### Task 6：完整消融训练

目标：训练 E0~E5。

产出：

```text
runs/bubble/E0_yolo11s_baseline
runs/bubble/E1_yolo11s_ssb
runs/bubble/E2_yolo11s_glrb
runs/bubble/E3_yolo11s_ssb_glrb
runs/bubble/E5_bubble_yolo11s_final
runs/bubble/experiment_summary.csv
```

验收：

```text
1. 所有实验均有 best.pt。
2. 所有实验均有 metrics。
3. 有消融表。
4. 有可视化对比图。
```

---

## 17. 最终论文可用表述

### 17.1 方法概述

```text
本文以 YOLO11s 为基础检测器，针对气泡图像中小目标密集、边界弱、重叠粘连以及高分辨率输入下跨尺度特征易失真的问题，设计了 Bubble-YOLO11s。该模型引入尺度变换增强模块以缓解上/下采样过程中的边界信息损失，引入全局-局部特征精炼模块以兼顾密集气泡场景下的上下文关系和局部轮廓细节，并结合小目标友好的 NWD 回归损失提升边界框定位稳定性。
```

### 17.2 SSB 表述

```text
尺度变换增强模块受 HSMD-YOLO 中 Scale Switch Block 启发，用于缓解特征金字塔上/下采样过程中的伪高频、噪声放大和弱边界破坏问题。考虑到工程稳定性，本文采用 shape-preserving 的局部卷积精炼结构，在保持特征图尺寸和通道数不变的前提下增强采样后的气泡边界表达。
```

### 17.3 GLRB 表述

```text
全局-局部特征精炼模块采用 MDTA 与 GDFN 组合。MDTA 在通道维度计算注意力，避免传统空间自注意力在高分辨率特征图上产生二次复杂度；GDFN 通过门控深度卷积前馈网络强化局部几何细节。该模块用于提升弱边界、重叠和密集气泡区域的特征可分性。
```

### 17.4 NWD 表述

```text
针对小气泡框对位置偏差高度敏感的问题，本文在原有 IoU 类边界框损失中融合 Normalized Gaussian Wasserstein Distance。该损失将边界框建模为二维高斯分布，通过中心距离和尺度差异共同约束预测框与真实框，从而提高小目标框回归稳定性。
```

---

## 18. 最终验收标准

最终交付应满足：

```text
1. YOLO11s baseline 可复现。
2. Bubble-YOLO11s 可训练、可验证、可推理。
3. 每个模块均有独立消融结果。
4. 代码中有 SSBRefine、GLRB、NWD loss 的清晰实现。
5. 每个模块有 shape test 和 forward test。
6. 有 debug subset 过拟合记录。
7. 有全量训练结果 summary。
8. 有固定测试图可视化对比。
9. 能解释 Precision、Recall、mAP@50、mAP@50-95、Params、FLOPs 的变化。
10. 不声称完全复现 HSMD-YOLO，只声明参考其问题建模和模块思想。
```

---

## 19. 最短可执行版本

如果时间极度紧张，只执行以下最短路径：

```text
1. 训练 YOLO11s baseline。
2. 实现 SSBRefine，插入 P3/P4/P5 Detect 前。
3. 实现 GLRB，只插 P3。
4. 训练 YOLO11s + SSBRefine + GLRB-P3。
5. 若时间允许，加入 NWD loss。
6. 做 E0 vs Final 的表格和可视化。
```

对应实验：

```text
E0: YOLO11s baseline
E1: YOLO11s + SSBRefine
E2: YOLO11s + SSBRefine + GLRB-P3
E3: YOLO11s + SSBRefine + GLRB-P3 + NWD
```

这是最现实、风险最低、论文也能讲清楚的路线。

