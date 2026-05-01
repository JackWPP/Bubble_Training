# 气泡检测 YOLO 训练工作区

气泡检测 YOLO 训练工作区。仓库仅保存可复现的数据处理脚本、训练脚本和数据集构建说明；原始图片、生成的 YOLO 数据集、模型权重和训练输出均作为本地资产管理，不纳入 git。

## 脚本概览

| 脚本 | 用途 |
| --- | --- |
| `07_build_integrated_dataset.py` | 从 COCO 数据构建 YOLO 训练数据集 |
| `scripts/train_experiment.py` | Bubble-YOLO11s 单实验训练、验证、汇总 |
| `scripts/run_nightly.py` | 按 E0-E5 顺序运行整夜消融训练 |
| `tools/check_model_forward.py` | 检查自定义 YOLO11s YAML 能否 forward |
| `tools/collect_results.py` | 汇总 Ultralytics 训练结果 |
| `tools/export_report.py` | 导出 Markdown 训练报告 |
| `03_train_yolo12n.py` | YOLO12n 微调训练 |
| `04_train_yolo12s.py` | YOLO12s 微调训练 |
| `05_predict.py` | 使用训练权重进行推理预测 |

## 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

依赖包括：`ultralytics>=8.3.0`、`opencv-python>=4.9.0`、`numpy>=1.26.0`。

## 目录约定

| 目录 | 说明 | 是否提交 |
| --- | --- | --- |
| `Dataset/` | 本地原始 COCO 数据集，结构为 `*/images/default` 与 `*/annotations/instances_default.json` | 否 |
| `yolo_dataset_grouped/` | 按组划分的 YOLO 数据集（推荐） | 否 |
| `yolo_dataset_integrated/` | 按图片随机划分的 YOLO 数据集（旧版） | 否 |
| `runs/` | Ultralytics 训练与预测输出 | 否 |

## 构建数据集

将整理好的 COCO 数据放入 `Dataset/` 后运行：

### 按组划分（推荐，用于论文实验）

```powershell
python .\07_build_integrated_dataset.py
```

默认输出到 `yolo_dataset_grouped/`。该模式按物理来源（视频、实验条件、采集序列）生成 `group_key`，确保同一组数据只出现在 train/val/test 中的一个子集，避免数据泄漏。

### 按图片随机划分（旧版）

```powershell
python .\07_build_integrated_dataset.py --split-mode source --output yolo_dataset_integrated
```

输出到 `yolo_dataset_integrated/`。

### 生成内容

- `{train,val,test}/images` — 训练/验证/测试图片
- `{train,val,test}/labels` — 对应 YOLO 标签
- `bubble.yaml` — Ultralytics 数据集配置
- `build_stats.json` — 构建统计
- `manifest.json` — 文件清单
- `quality_preview/` — 质量预览
- `DATASET_BUILD_REPORT.md` 或 `DATASET_GROUPED_BUILD_REPORT.md` — 构建报告

### 处理策略

- 按组划分：`train/val/test = 75/10/15`；按图片划分：`train/val/test = 80/15/5`，固定种子 `42`。
- 大图使用 `640×640` 滑窗切片，步幅 `480`。
- 小图保持比例缩放并 padding 到 `640×640`。
- 仅对训练集做离线增强，验证集和测试集不增强。
- 所有目标统一为单类 `bubble`（类别编号 `0`）。

## Bubble-YOLO11s 训练

训练策略和实验矩阵见 `BUBBLE_YOLO_TRAINING_PLAN.md`。新实验统一使用 YOLO11s 与 grouped 数据集：

```powershell
python .\07_build_integrated_dataset.py --split-mode balanced-v2 --output yolo_dataset_balanced_v2
python .\tools\validate_balanced_v2_dataset.py
python .\tools\make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python .\tools\validate_train_dev_split.py
python .\tools\validate_grouped_dataset.py
python -m pytest .\tests\test_bubble_modules.py
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_final.yaml
```

本机冒烟训练：

```powershell
python .\scripts\train_experiment.py --exp E0 --preset smoke --device 0 --exist-ok
python .\scripts\train_experiment.py --exp E3 --preset smoke --device 0 --exist-ok
```

双 V100 服务器整夜训练：

```bash
python tools/make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python tools/validate_train_dev_split.py
python scripts/train_experiment.py --exp BV2S --device 0,1 --exist-ok
```

压缩矩阵：

```bash
python scripts/run_nightly.py --preset full_conservative --device 0,1 --compressed --resume-missing
```

训练输出默认写入 `runs/bubble/`，训练结束后可手动汇总：

```powershell
python .\tools\collect_results.py --project runs\bubble
python .\tools\export_report.py --project runs\bubble
```

## 历史 YOLO12 训练脚本

训练脚本默认使用 `yolo_dataset_grouped/bubble.yaml`：

```powershell
python .\03_train_yolo12n.py
python .\04_train_yolo12s.py
```

权重文件（`*.pt`）不纳入 git。可通过环境变量覆盖默认配置：

```powershell
$env:BUBBLE_PRETRAINED_WEIGHTS="G:\Bubble_Train\yolo12n.pt"
$env:BUBBLE_DATA_CONFIG="G:\Bubble_Train\yolo_dataset_grouped\bubble.yaml"
$env:BUBBLE_DEVICE="0"
python .\03_train_yolo12n.py
```

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BUBBLE_PRETRAINED_WEIGHTS` | 项目根目录下的 `yolo26n.pt` / `yolo12s.pt` | 预训练权重路径 |
| `BUBBLE_DATA_CONFIG` | `yolo_dataset_grouped/bubble.yaml` | 数据集配置文件路径 |
| `BUBBLE_PROJECT_DIR` | `runs/` | 训练输出目录 |
| `BUBBLE_DEVICE` | `0` | GPU 设备编号 |

新训练入口还支持：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BUBBLE_PRETRAINED_WEIGHTS` | `yolo11s.pt` | YOLO11s 预训练权重路径；服务器无网时建议显式设置 |
| `BUBBLE_DEVICE` | `configs/train/*.yaml` 中的值 | GPU 设备编号，双卡可用 `0,1` |

## 推理预测

```powershell
python .\05_predict.py <模型路径> <图片目录> <输出目录>
```

也支持环境变量配置：

```powershell
$env:BUBBLE_MODEL_PATH="G:\Bubble_Train\runs\yolo12n_finetune\weights\best.pt"
$env:BUBBLE_SOURCE_DIR="G:\Bubble_Train\yolo_dataset_grouped\val\images"
$env:BUBBLE_OUTPUT_DIR="G:\Bubble_Train\runs\predict_val"
python .\05_predict.py
```

## 数据集报告

详细的数据集构建过程、切片策略、增强策略、样本分布和已知限制请参阅：

- `DATASET_GROUPED_BUILD_REPORT.md` — 按组划分版本
- `DATASET_BUILD_REPORT.md` — 按图片划分版本

## Git 规则

**提交内容：**

- 数据处理和训练脚本
- 依赖文件（`requirements.txt`）
- README 和数据集报告

**不提交内容：**

- 原始数据集和生成数据集
- 模型权重
- 训练输出
- 本地缓存和临时分析结果
