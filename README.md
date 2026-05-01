# Bubble Train

气泡检测 YOLO 训练工作区。仓库只保存可复现的数据处理、训练脚本和数据集构建说明；原始图片、生成后的 YOLO 数据集、模型权重和训练输出都作为本地资产管理，不进入 git。

## 目录约定

- `Dataset/`：本地原始 COCO 数据集来源，结构为 `*/images/default` 与 `*/annotations/instances_default.json`。
- `yolo_dataset_integrated/`：由构建脚本生成的综合 YOLO 数据集，本地训练使用，不提交。
- `runs/`：Ultralytics 训练与验证输出，不提交。
- `DATASET_BUILD_REPORT.md`：综合数据集的构建方法、统计特性和样本分布说明。

## 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 构建综合数据集

将整理好的 COCO 数据放入 `Dataset/` 后运行：

```powershell
python .\07_build_integrated_dataset.py
```

脚本会生成：

- `yolo_dataset_integrated/train|val|test/images`
- `yolo_dataset_integrated/train|val|test/labels`
- `yolo_dataset_integrated/bubble.yaml`
- `yolo_dataset_integrated/build_stats.json`
- `yolo_dataset_integrated/manifest.json`
- `yolo_dataset_integrated/quality_preview/`
- `DATASET_BUILD_REPORT.md`

默认策略：

- 原图随机拆分为 `train/val/test = 80/15/5`，固定种子 `42`。
- 大图使用 `640x640` 滑窗，stride 为 `480`。
- 小图保持比例缩放并 padding 到 `640x640`。
- 只对训练集做离线增强，验证集和测试集保持未增强。
- 所有目标统一为单类 `bubble`。

## 训练

训练脚本中的 `DATA_CONFIG` 应指向：

```text
G:\Bubble_Train\yolo_dataset_integrated\bubble.yaml
```

例如：

```powershell
python .\03_train_yolo12n.py
```

权重文件如 `*.pt` 不纳入 git。需要复现实验时，请在本地准备对应预训练权重；也可以通过环境变量覆盖默认路径：

```powershell
$env:BUBBLE_PRETRAINED_WEIGHTS="G:\Bubble_Train\yolo26n.pt"
$env:BUBBLE_DATA_CONFIG="G:\Bubble_Train\yolo_dataset_integrated\bubble.yaml"
$env:BUBBLE_DEVICE="0"
python .\03_train_yolo12n.py
```

预测脚本同样支持命令行参数或环境变量：

```powershell
python .\05_predict.py <model_path> <source_dir> <output_dir>
```

## 数据集说明

综合数据集的构建过程、切片策略、增强策略、样本分布和已知限制见：

- `DATASET_BUILD_REPORT.md`

该报告说明了当前数据集不是追求来源数量的机械均匀，而是面向气泡检测任务提高了密集、小目标和复杂场景的训练覆盖。

## Git 规则

仓库只提交：

- 数据处理和训练脚本
- 依赖文件
- README 和数据集报告

不提交：

- 原始数据集和生成数据集
- 模型权重
- 训练输出
- 本地缓存和临时分析结果
