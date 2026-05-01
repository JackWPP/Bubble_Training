# Bubble-YOLO Balanced V2 Handoff

生成时间：2026-05-02  
接手目标：继续基于 `yolo_dataset_balanced_v2` 跑 YOLO11s 主 baseline，并在稳定后进入 SSB/GLRB/NWD 消融。

## 1. 当前结论

本轮已经确认：旧 `yolo_dataset_grouped` 不适合作为主训练/主消融数据集。

原因不是单纯学习率或训练策略问题，而是旧 split 把若干关键 domain 几乎完整放在 val/test，导致 train 无法覆盖：

- `big_fengchao`
- `bubble_1`
- `bubble_pad`
- 部分 `bubble_fc` / `job_13` 工况

旧 grouped split 后续仍保留，但角色改为 OOD stress test，不再用于主 baseline 选择，也不作为 E1-E5 消融主数据集。

新的主实验数据集已经确定为：

```text
yolo_dataset_balanced_v2/bubble.yaml
```

## 2. 本轮已完成改动

### 2.1 数据集构建

修改文件：

```text
07_build_integrated_dataset.py
```

新增能力：

- 新增 `--split-mode balanced-v2`，兼容 `balanced_v2`。
- 新增 `split_sources_balanced_v2()`。
- 新增 `split_source_key_bucket()`。
- split 逻辑为 `source -> group_key -> source_key` 分层切分。
- 同一 `source_key` 不跨 train/val/test。
- `balanced-v2` 不强制 `group_key` 完全隔离，因为目标是主训练集覆盖所有来源，不是整源 OOD。
- `balanced-v2` 下 train 离线增强降强度：
  - 保留 `hflip`
  - 保留亮度/对比度/HSV/轻噪声等 photometric 增强
  - 关闭 `copy_paste`
  - 不再做 vflip/rot90/rot180/rot270
- 新增 Balanced V2 报告输出：
  - `yolo_dataset_balanced_v2/DATASET_BUILD_REPORT.md`
  - `DATASET_BUILD_REPORT.md`
  - `DATASET_BALANCED_V2_BUILD_REPORT.md`

构建命令：

```powershell
python .\07_build_integrated_dataset.py --split-mode balanced-v2 --output yolo_dataset_balanced_v2
```

### 2.2 数据集验证

新增文件：

```text
tools/validate_balanced_v2_dataset.py
```

验证内容：

- `split_mode == balanced-v2`
- 所有主要 source 都出现在 train/val/test
- `source_key` 不跨 split
- val/test 没有增强样本
- train/val/test 图像目录存在

验证命令：

```powershell
python .\tools\validate_balanced_v2_dataset.py
```

当前验证结果：

```text
balanced-v2 validation ok
train: images=2736 boxes=75744 avg=27.68 sources=[
  '20+40',
  '60+80',
  'big_fengchao',
  'bubble_1',
  'bubble_fc',
  'bubble_pad',
  'job_13_dataset_2026_04_30_19_34_23_coco 1.0'
]
val: images=92 boxes=2877 avg=31.27 sources=[
  '20+40',
  '60+80',
  'big_fengchao',
  'bubble_1',
  'bubble_fc',
  'bubble_pad',
  'job_13_dataset_2026_04_30_19_34_23_coco 1.0'
]
test: images=92 boxes=2663 avg=28.95 sources=[
  '20+40',
  '60+80',
  'big_fengchao',
  'bubble_1',
  'bubble_fc',
  'bubble_pad',
  'job_13_dataset_2026_04_30_19_34_23_coco 1.0'
]
```

### 2.3 训练配置

新增文件：

```text
configs/train/balanced_v2_nano_smoke.yaml
configs/train/balanced_v2_baseline.yaml
```

修改文件：

```text
configs/train/experiments.yaml
```

新增实验：

```yaml
BV2N:
  name: BV2N_yolo11n_balanced_v2_smoke
  model: yolo11n.pt
  modules: balanced-v2-nano-smoke
  use_nwd: false
  train_config: configs/train/balanced_v2_nano_smoke.yaml

BV2S:
  name: BV2S_yolo11s_balanced_v2_baseline
  model: yolo11s.pt
  modules: balanced-v2-yolo11s-baseline
  use_nwd: false
  train_config: configs/train/balanced_v2_baseline.yaml
```

`BV2S` 默认关键训练参数：

```yaml
data: yolo_dataset_balanced_v2/bubble.yaml
eval_data: yolo_dataset_grouped/bubble.yaml
project: runs/bubble_balanced_v2
imgsz: 640
epochs: 100
batch: 16
workers: 8
device: 0,1
seed: 42
patience: 50
optimizer: SGD
lr0: 0.003
momentum: 0.937
weight_decay: 0.0005
cos_lr: true
save_period: 5
mosaic: 0.1
mixup: 0.0
copy_paste: 0.0
translate: 0.03
scale: 0.1
```

### 2.4 训练入口和报告工具

修改文件：

```text
scripts/train_experiment.py
tools/collect_results.py
tools/export_report.py
```

`scripts/train_experiment.py` 改动：

- 支持实验项自带 `train_config` 时，不再自动给 run name 追加 `_smoke`。
- 修正 `pretrained_weight` 记录：
  - `model=yolo11n.pt` 时记录为 `yolo11n.pt`
  - `model=yolo11s.pt` 时记录为 `yolo11s.pt`
  - 自定义 YAML 时记录 `--weights` 或 `BUBBLE_PRETRAINED_WEIGHTS`

`tools/collect_results.py` 改动：

- 汇总字段新增：
  - `pretrained_weight`
  - `data_config`
  - `official_eval_data_config`
- 读取 summary 时兼容 UTF-8 BOM。

`tools/export_report.py` 改动：

- 报告顶部区分：
  - Training dataset
  - Official eval dataset
- 不再硬编码 `yolo_dataset_grouped/bubble.yaml` 为 Dataset。

### 2.5 文档和项目规则

修改文件：

```text
.gitignore
README.md
BUBBLE_YOLO_TRAINING_PLAN.md
AGENTS.md
```

要点：

- `.gitignore` 已加入 `yolo_dataset_balanced_v2/`。
- 文档中主训练路径切换为 Balanced V2。
- `yolo_dataset_grouped` 明确为 OOD stress test。
- 推荐服务器训练命令改为 `BV2S`。

## 3. 本轮已跑验证

### 3.1 代码编译

已跑：

```powershell
python -m py_compile 07_build_integrated_dataset.py scripts\train_experiment.py tools\collect_results.py tools\export_report.py tools\validate_balanced_v2_dataset.py
python -m compileall 07_build_integrated_dataset.py scripts tools tests
```

结果：通过。

### 3.2 数据集构建和校验

已跑：

```powershell
python .\07_build_integrated_dataset.py --split-mode balanced-v2 --output yolo_dataset_balanced_v2
python .\tools\validate_balanced_v2_dataset.py
```

结果：通过。

关键验收：

- `source_key_split_leakage_count == 0`
- `augmented_val_test_count == 0`
- 每个主要 source 都有 train/val/test 样本
- `group_split_leakage_count` 在 balanced-v2 下可能非 0，这是预期行为，因为该模式不做整 group OOD 隔离

### 3.3 Ultralytics 数据读取

已验证 `yolo_dataset_balanced_v2/bubble.yaml` 可被 Ultralytics 读取。

验证命令：

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; d=check_det_dataset('yolo_dataset_balanced_v2/bubble.yaml'); print(d['train']); print(d['val']); print(d['test'])"
```

结果：通过。

### 3.4 本地 nano smoke 训练

本地环境：

- GPU：RTX 4060 Laptop GPU
- Ultralytics：8.4.21
- Torch：2.9.1+cu130
- Python：3.12.6

已跑命令：

```powershell
python scripts\train_experiment.py --exp BV2N --epochs 5 --batch 4 --workers 0 --device 0 --exist-ok --skip-predict
```

训练结果路径：

```text
runs/bubble_balanced_v2_smoke/BV2N_yolo11n_balanced_v2_smoke
```

summary：

```text
runs/bubble_balanced_v2_smoke/BV2N_yolo11n_balanced_v2_smoke/summary.json
```

汇总报告：

```text
runs/bubble_balanced_v2_smoke/TRAINING_REPORT.md
```

5 epoch 曲线：

```text
epoch 1: mAP50-95 = 0.32799
epoch 2: mAP50-95 = 0.36062
epoch 3: mAP50-95 = 0.36251
epoch 4: mAP50-95 = 0.40112
epoch 5: mAP50-95 = 0.39957
```

best epoch：4。

这已经解决了旧 baseline 中 “best 卡在 epoch 1-2，随后验证持续下跌” 的核心症状。

自动评估结果：

```text
best.pt selection val mAP50-95: 0.4018
best.pt old grouped official val mAP50-95: 0.3913
best.pt old grouped official test mAP50-95: 0.4921

last.pt selection val mAP50-95: 0.4000
last.pt old grouped official val mAP50-95: 0.3913
last.pt old grouped official test mAP50-95: 0.5007
```

注意：这是 `yolo11n.pt` 5 epoch smoke，不是正式 baseline。它只证明数据集和训练流程方向正确。

## 4. 接下来最直接的任务

### 4.1 在双 V100 服务器跑 YOLO11s Balanced V2 baseline

优先执行：

```bash
python scripts/train_experiment.py --exp BV2S --device 0,1 --exist-ok
```

预计输出：

```text
runs/bubble_balanced_v2/BV2S_yolo11s_balanced_v2_baseline
```

训练结束后执行：

```bash
python tools/collect_results.py --project runs/bubble_balanced_v2
python tools/export_report.py --project runs/bubble_balanced_v2
```

检查：

```text
runs/bubble_balanced_v2/experiment_summary.csv
runs/bubble_balanced_v2/experiment_summary.json
runs/bubble_balanced_v2/TRAINING_REPORT.md
```

### 4.2 判定 BV2S 是否稳定

至少看这些点：

- best epoch 不应集中在 epoch 1-2。
- balanced val `mAP50-95` 不应长期单边下跌。
- train loss 下降时，val metrics 应整体可维持或提升。
- best.pt 和 last.pt 都要报告，尤其当 official test 差异明显时。
- old grouped val/test 只作为 OOD 结果，不作为 checkpoint 选择依据。

建议最低接受标准：

- Balanced V2 test 上建立稳定 YOLO11s baseline。
- old grouped OOD test 结果不崩。
- best.pt 与 last.pt 不出现极端分歧；若分歧明显，报告必须同时保留两者。

### 4.3 BV2S 通过后再进入消融

建议顺序：

1. `BV2S`：YOLO11s Balanced V2 baseline
2. `E1`：YOLO11s + SSBRefine
3. `E3`：YOLO11s + SSBRefine + GLRB-P3
4. `E5`：Bubble-YOLO11s + NWD loss

先不要急着跑全矩阵 E0-E5。当前最关键是把 Balanced V2 上的 YOLO11s baseline 定稳。

## 5. 重要注意事项

### 5.1 不要回退到旧 grouped 主训练逻辑

`yolo_dataset_grouped` 的新角色是：

```text
OOD stress test only
```

它不应再用于：

- 主 baseline checkpoint selection
- 主论文消融 split
- 训练过程 early stopping

### 5.2 不要删除旧 grouped 数据集

旧 grouped 仍有价值：

- 可作为跨域泛化附加实验。
- 可帮助证明 Balanced V2 是主实验数据集，grouped 是更严格的 OOD 评估。

### 5.3 不要把数据集和 runs 提交到 git

这些路径应保持未跟踪或 ignored：

```text
yolo_dataset_balanced_v2/
yolo_dataset_grouped/
runs/
*.pt
```

当前 `.gitignore` 已加入：

```text
yolo_dataset_balanced_v2/
```

### 5.4 当前工作树是脏的

本轮有大量已修改/新增文件。接手智能体不要误用 `git reset --hard` 或 `git checkout --`。

当前重要新增文件：

```text
HANDOFF_BALANCED_V2.md
DATASET_BUILD_REPORT.md
DATASET_BALANCED_V2_BUILD_REPORT.md
configs/train/balanced_v2_baseline.yaml
configs/train/balanced_v2_nano_smoke.yaml
tools/validate_balanced_v2_dataset.py
```

当前重要修改文件：

```text
.gitignore
07_build_integrated_dataset.py
AGENTS.md
BUBBLE_YOLO_TRAINING_PLAN.md
README.md
configs/train/experiments.yaml
scripts/train_experiment.py
tools/collect_results.py
tools/export_report.py
```

## 6. 常用命令清单

重建 Balanced V2：

```powershell
python .\07_build_integrated_dataset.py --split-mode balanced-v2 --output yolo_dataset_balanced_v2
```

校验 Balanced V2：

```powershell
python .\tools\validate_balanced_v2_dataset.py
```

本地 nano smoke：

```powershell
python scripts\train_experiment.py --exp BV2N --epochs 5 --batch 4 --workers 0 --device 0 --exist-ok --skip-predict
```

服务器 YOLO11s baseline：

```bash
python scripts/train_experiment.py --exp BV2S --device 0,1 --exist-ok
```

汇总结果：

```bash
python tools/collect_results.py --project runs/bubble_balanced_v2
python tools/export_report.py --project runs/bubble_balanced_v2
```

检查编译：

```powershell
python -m compileall 07_build_integrated_dataset.py scripts tools tests
```

## 7. 接手建议

新智能体接手后，不要重新争论是否重整数据集。当前证据已经足够：

- 旧 grouped baseline best epoch 异常。
- B1/B2 conservative/freeze 没有根治。
- Balanced V2 nano smoke 曲线正常，best epoch 出现在 epoch 4。

直接推进：

1. 确认服务器已有 `yolo_dataset_balanced_v2` 或重建。
2. 跑 `BV2S`。
3. 生成 `TRAINING_REPORT.md`。
4. 根据 BV2S 结果决定是否进入 E1/E3/E5。

