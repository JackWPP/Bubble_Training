# SAM3 推理 — 服务器部署 Handoff

> 生成时间: 2026-05-04
> 状态: generate_masks.py 待完整运行（20+40 & 60+80 已完成，其余 5 个源待处理）
> GPU 需求: NVIDIA GPU >= 12GB VRAM（推荐），最低 8GB（会慢）

## 一、背景

Bubble_Train 项目目前有 7 个 COCO bbox-only 数据源（170 张原图，9,524 个标注框），需要利用 SAM3 将 bbox 升级为高质量分割 mask。

工作流程：
```
COCO bbox-only → SAM3 推理 → COCO bbox+polygon → build_dataset.py → YOLO-seg 数据集 → YOLO-seg 训练
```

## 二、文件说明

```
segmentation/
├── HANDOFF_SAM3_SERVER.md          # 本文件
├── generate_masks.py               # ★ 核心脚本：SAM3 推理生成分割标注
├── test_sam3_wsl.py                #   验证脚本：SAM3 加载 + 单图测试
├── test_sam3_concept.py            #   验证脚本：概念匹配测试
├── test_sam3_resize.py             #   验证脚本：大图缩放测试
└── debug_single_image.py           #   调试脚本：单图各阶段耗时分析
```

依赖的主项目文件（需在同仓库中）：
```
Dataset/*/annotations/instances_default.json     # 输入：COCO bbox 标注
Dataset/*/images/default/*.jpg                    # 输入：原始图片
ultralytics_custom/                               # 不需要，generate_masks.py 不依赖此目录
```

## 三、环境配置

### 3.1 系统要求

| 组件 | 要求 |
|------|------|
| Python | >= 3.12 |
| CUDA | >= 12.6 |
| GPU VRAM | >= 12GB（推荐）/ 8GB（可跑，较慢） |
| 磁盘 | ~4GB（SAM3 权重） + ~5MB（生成的 JSON 标注） |

### 3.2 安装依赖

```bash
# 创建虚拟环境
python3 -m venv sam3_env
source sam3_env/bin/activate  # Linux
# sam3_env\Scripts\activate   # Windows

# PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# 如果 CUDA 版本不同，参考: https://pytorch.org/get-started/locally/

# 其他依赖（已提供 requirements_sam3.txt）
pip install -r requirements_sam3.txt
```

### 3.3 SAM3 权重

从 HuggingFace 下载 SAM3 权重，放到服务器上后设置路径：

```bash
# 权重文件结构（HuggingFace 格式）：
/path/to/sam3/
├── model.safetensors      # 3.4GB
├── config.json
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
└── sam3.pt                # 可选：Ultralytics 格式

# 修改 generate_masks.py 中的 MODEL_PATH：
# MODEL_PATH = "/path/to/sam3"
```

**注意**：SAM3 权重是 HuggingFace gated model，需要先在 huggingface.co/facebook/sam3 申请访问权限，然后用 `huggingface-cli login` 登录后下载。
如果已有 `sam3.pt` 文件（Ultralytics 格式），也可以放在同一目录下作为备选。

## 四、运行步骤

### Step 1: 验证环境

```bash
cd /path/to/Bubble_Train/segmentation
python test_sam3_wsl.py
```

预期输出：
```
PyTorch 2.x.x+cuXXX, CUDA: True
GPU: NVIDIA GeForce XXX, VRAM: XX.X GB
SAM3 loaded successfully.
...
SAM3 verification PASSED.
```

同时在 `segmentation/test_sam3_output.jpg` 生成可视化结果。

### Step 2: 单图耗时测试（可选）

```bash
python debug_single_image.py
```

这会处理 `big_fengchao` 的第一张图（1920×1080, 313 个标注框），输出各阶段耗时。

**注意**：首次运行会触发 CUDA kernel 编译，模型加载约 60-80s。之后每次前向推理预计：
- RTX 4090 (24GB): ~2-5 秒/图
- RTX 4060 (8GB): ~25-40 秒/次前向（较慢）

### Step 3: 全量推理

```bash
# 先预览数据源
python generate_masks.py --dry-run

# 单源测试
python generate_masks.py --source "20+40"

# 全量运行（跳过已完成的源）
python generate_masks.py
```

**断点续跑**：脚本会自动检测 `Dataset/*/annotations/instances_default_segmented.json` 是否存在，已完成的源会被跳过。当前进度：

| 数据源 | 图片数 | 标注数 | 状态 |
|--------|--------|--------|------|
| 20+40 | 66 | 2,233 | ✅ 已完成 |
| 60+80 | 47 | 442 | ✅ 已完成 |
| big_fengchao | 6 | 1,460 | ⏳ 待处理 |
| bubble_1 | 3 | 590 | ⏳ 待处理 |
| bubble_fc | 12 | 602 | ⏳ 待处理 |
| bubble_pad | 3 | 1,484 | ⏳ 待处理 |
| job_13 | 33 | 2,713 | ⏳ 待处理 |

### Step 4: 验证输出

```bash
# 检查某个源的分割标注覆盖
python3 -c "
import json
d = json.load(open('Dataset/20+40/annotations/instances_default_segmented.json'))
segs = [a for a in d['annotations'] if a.get('segmentation') and a['segmentation']!=[[]]]
print(f'{len(segs)}/{len(d[\"annotations\"])} have segmentation')
"
```

## 五、输出格式

`instances_default_segmented.json` 在原有 COCO JSON 基础上，为每个 annotation 添加了 `segmentation` 字段：

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [22.91, 358.87, 11.29, 10.23],
      "segmentation": [[24.0, 359.0, 23.0, 360.0, 22.0, 366.0, ...]]
    }
  ]
}
```

Polygon 格式为 COCO 标准：`[[x1, y1, x2, y2, ..., xn, yn]]`（绝对像素坐标）。

## 六、后续步骤（回到 Windows）

SAM3 推理完成后，在 Windows 端运行：

```bash
# 1. 构建 YOLO-seg 数据集
cd G:\Bubble_Train
python segmentation/build_dataset.py

# 2. Smoke train (验证 pipeline)
python segmentation/scripts/train_seg.py \
    --model segmentation/configs/models/bubble_yolo11s_seg.yaml \
    --cfg segmentation/configs/train/seg_smoke.yaml

# 3. Full train (100 epoch)
python segmentation/scripts/train_seg.py \
    --model segmentation/configs/models/bubble_yolo11s_seg.yaml \
    --cfg segmentation/configs/train/seg_full.yaml \
    --name bubble_seg_baseline

# 4. +P3LCRefine (最佳检测改进迁移)
python segmentation/scripts/train_seg.py \
    --model segmentation/configs/models/bubble_yolo11s_seg_p3lc.yaml \
    --cfg segmentation/configs/train/seg_full.yaml \
    --name bubble_seg_p3lc \
    --nwd 0.05
```

## 七、关键参数说明

`generate_masks.py` 中的可调参数：

```python
MAX_IMAGE_DIM = 1024      # 大图缩放目标（长边最大像素），越大越精确但越慢
MAX_CONCEPT_BOXES = 20    # 概念模式每次最多使用的 bbox 数量
IOU_THRESHOLD = 0.3       # mask 匹配的 IoU 阈值
CONF_THRESHOLD = 0.3      # SAM3 置信度阈值
```

**服务器性能调优建议**：
- 如果 GPU >= 16GB VRAM：可将 `MAX_IMAGE_DIM` 提高到 1536 或保持原图
- 如果 GPU >= 24GB VRAM：可将 `MAX_CONCEPT_BOXES` 提高到 50-100 加速推理
- 如果遇到 OOM：降低 `MAX_IMAGE_DIM` 到 768 或 640

## 八、已知问题

1. **SAM3 概念分割召回率低**：当 bbox 数量多且形态差异大时，概念模式可能只匹配少量气泡，其余走 fallback 批量推理，总体耗时增加。
2. **首次推理慢**：CUDA kernel JIT 编译导致首次模型加载 ~60-80s，后续正常。
3. **VRAM 限制**：8GB GPU 上 SAM3 使用 ~7.7GB，几乎没有余量。推荐 12GB+ GPU。

---

### 联系方式

有问题请在 Bubble_Train 主仓库提 issue。
