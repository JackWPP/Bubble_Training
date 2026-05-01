# Bubble-YOLO11s 训练策略落地计划

## 目标

本项目正式训练统一使用 `yolo_dataset_grouped/bubble.yaml`。该数据集按物理源头和采集条件隔离 train/val/test，避免相邻帧、同实验条件和同源切片造成指标虚高。`yolo_dataset_integrated` 只作为历史开发数据集，不作为论文最终泛化评估依据。

训练主线固定为 YOLO11s，不切换到 YOLO12 或 YOLO26。现有 `03_train_yolo12n.py`、`04_train_yolo12s.py` 保留为历史脚本，新实验统一从 `scripts/train_experiment.py` 和 `scripts/run_nightly.py` 进入。

## 实验矩阵

实验定义维护在 `configs/train/experiments.yaml`：

| 实验 | 模型 | 目的 |
| --- | --- | --- |
| E0 | YOLO11s baseline | 建立可复现主基线 |
| E1 | YOLO11s + SSBRefine-P3 | 验证采样后局部边界精炼 |
| E2 | YOLO11s + GLRB-P3 | 验证全局-局部注意力精炼 |
| E3 | YOLO11s + SSBRefine-P3 + GLRB-P3 | 验证低风险组合效果 |
| E4 | YOLO11s + SSBRefine/GLRB-P3/P4/P5 | 验证多尺度插入收益 |
| E5 | Bubble-YOLO11s + NWD loss | 最终候选模型 |

时间不足时执行压缩矩阵：E0、E1、E3、E5。

## 训练配置

配置文件位于 `configs/train/`：

- `smoke.yaml`：本机 RTX 4060 或其他单卡冒烟，默认 2 epoch、batch 2。
- `debug_overfit.yaml`：配合 `tools/make_debug_subset.py` 做小样本过拟合检查。
- `full.yaml`：双 V100 正式训练，默认 200 epoch、batch 16、device `0,1`。
- `full_conservative.yaml`：修正 baseline 后的默认正式配置，使用 train-domain dev-val 选择 checkpoint，并在训练后评估 official grouped val/test。
- `full_conservative_freeze.yaml`：B2 备用配置，冻结前 10 层做低学习率微调。

默认增强策略保持克制，因为 grouped 数据集训练集已有离线增强：

```yaml
mosaic: 0.3
mixup: 0.0
copy_paste: 0.0
degrees: 0.0
translate: 0.05
scale: 0.2
hsv_h: 0.005
hsv_s: 0.2
hsv_v: 0.2
fliplr: 0.5
flipud: 0.0
```

## 常用命令

本机环境检查：

```powershell
python .\tools\make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python .\tools\validate_train_dev_split.py
python .\tools\validate_grouped_dataset.py
python -m pytest .\tests\test_bubble_modules.py
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_ssb_glrb_p3.yaml
```

生成 debug subset：

```powershell
python .\tools\make_debug_subset.py --clean --train 40 --val 10
python .\scripts\train_experiment.py --exp E3 --preset debug_overfit --device 0 --exist-ok
```

本机冒烟：

```powershell
python .\scripts\train_experiment.py --exp E0 --preset smoke --device 0 --exist-ok
python .\scripts\train_experiment.py --exp E3 --preset smoke --device 0 --exist-ok
```

双 V100 服务器整夜训练：

```bash
python tools/make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python tools/validate_train_dev_split.py
python scripts/run_nightly.py --preset full_conservative --device 0,1 --baseline-fix --resume-missing
```

只跑压缩矩阵：

```bash
python scripts/run_nightly.py --preset full_conservative --device 0,1 --compressed --resume-missing
```

baseline 修复矩阵：

- `B0_yolo11s_pretrained_eval`：只评估 `yolo11s.pt`，不训练。
- `B1_yolo11s_conservative`：低学习率、弱增强 baseline。
- `B2_yolo11s_conservative_freeze`：冻结前 10 层的备用 baseline。

服务器无网时，提前放置权重并指定：

```bash
export BUBBLE_PRETRAINED_WEIGHTS=/path/to/yolo11s.pt
python scripts/run_nightly.py --preset full --device 0,1 --weights "$BUBBLE_PRETRAINED_WEIGHTS" --resume-missing
```

## 自定义模块接入

仓库内的 `ultralytics_custom/` 提供：

- `SSBRefine`：shape-preserving 局部边界精炼模块。
- `GLRB`：Restormer 思路的 MDTA + GDFN 全局-局部精炼模块。
- `bubble_loss.py`：NWD loss 计算与 Ultralytics `BboxLoss` 运行时 patch。
- `register.py`：在训练脚本运行时向 Ultralytics YAML parser 注册自定义模块。

所有仓库脚本都会运行时注册，不需要修改 site-packages。若需要直接使用 `yolo train model=configs/models/*.yaml` CLI，可在服务器上 clone editable Ultralytics 后运行：

```bash
git clone https://github.com/ultralytics/ultralytics.git ultralytics_src
python tools/patch_ultralytics_source.py --ultralytics-src ultralytics_src
pip install -e ultralytics_src
```

## 结果归档

每个实验输出：

- `runs/bubble/<experiment>/weights/best.pt`
- `runs/bubble/<experiment>/weights/last.pt`
- `runs/bubble/<experiment>/results.csv`
- `runs/bubble/<experiment>/summary.json`
- `runs/summary/<experiment>.json`

整夜训练结束后自动生成：

- `runs/bubble/experiment_summary.csv`
- `runs/bubble/experiment_summary.json`
- `runs/bubble/TRAINING_REPORT.md`

也可以手动生成：

```bash
python tools/collect_results.py --project runs/bubble
python tools/export_report.py --project runs/bubble
```

## 验收门槛

进入正式 full 训练前必须满足：

1. `tools/validate_grouped_dataset.py` 通过。
2. `tools/validate_train_dev_split.py` 通过。
3. `tests/test_bubble_modules.py` 通过。
4. 所有计划训练的 `configs/models/*.yaml` 通过 `tools/check_model_forward.py`。
5. B0/B1/B2 修复矩阵跑完，并确认 official test 指标没有被 dev-val 选择过程污染。
6. 每个自定义模块至少通过一次 smoke 或 debug overfit。

最终报告必须同时解释 Precision、Recall、mAP@50、mAP@50-95、Params、FLOPs 的变化；若模块使某些指标下降但改善密集/弱边界场景，也必须如实记录。

## Baseline 异常修复原则

`yolo_dataset_grouped/val` 和 `test` 是论文级严格泛化评估集，不再用于 early stopping。训练过程中的 checkpoint 选择改用 `configs/data/bubble_train_dev.yaml` 的 dev-val；summary 和报告中必须分开记录：

- `selection_val_metrics`：train-domain dev-val，仅用于选择 checkpoint。
- `official_val_metrics`：grouped val，仅用于训练后泛化评估。
- `official_test_metrics`：grouped test，作为最终 baseline 对比重点。
- `checkpoint_metrics.best/last`：best.pt 和 last.pt 的分开评估结果。
