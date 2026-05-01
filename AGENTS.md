# Bubble_Train Agent Guide

本文件是 `g:\Bubble_Train` 的项目级协作约定。它补充全局 Codex/OMX 指令，不覆盖全局安全、权限、验证和提交规则。

## 项目目标

- 正式数据集固定为 `yolo_dataset_grouped/bubble.yaml`。
- 正式训练主线固定为 YOLO11s 与 Bubble-YOLO11s 消融实验。
- 不把 `Dataset/`、`yolo_dataset_*`、`runs/`、权重文件或预测图片提交到 git。
- 不把 YOLO12/YOLO26 脚本作为新实验主入口；旧脚本只保留历史参考。

## 协作分工

- 训练入口维护者：优先修改 `scripts/train_experiment.py`、`scripts/run_nightly.py`、`configs/train/*.yaml`。
- 模型结构维护者：优先修改 `ultralytics_custom/bubble_modules.py` 和 `configs/models/*.yaml`。
- 损失函数维护者：优先修改 `ultralytics_custom/bubble_loss.py`，不得同时改 NMS 或 assignment。
- 结果分析维护者：优先修改 `tools/collect_results.py`、`tools/export_report.py`。
- 数据集维护者：优先修改 `07_build_integrated_dataset.py` 和数据集报告，不在训练阶段临时改变 split。

## 训练约束

- 每个正式实验必须先 smoke，再 full。
- 每个自定义 YAML 必须先 forward check，再训练。
- E0 baseline 不稳定时，不进入 E1-E5。
- NWD 只作为 box loss 融合项；当前阶段不实现 NWD-NMS、NWD assignment、BEMAF、P2 head。
- 若双卡 DDP 与自定义模块冲突，先用单卡验证同一配置，再切换到 `torch.distributed.run` 或 editable Ultralytics patch。

## 推荐命令

```powershell
python .\tools\validate_grouped_dataset.py
python .\tools\make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python .\tools\validate_train_dev_split.py
python -m pytest .\tests\test_bubble_modules.py
python .\tools\check_model_forward.py --model configs\models\bubble_yolo11s_final.yaml
python .\scripts\train_experiment.py --exp E0 --preset smoke --device 0 --exist-ok
```

服务器整夜训练：

```bash
python tools/make_train_dev_split.py --ratio 0.15 --min-images 50 --seed 42
python tools/validate_train_dev_split.py
python scripts/run_nightly.py --preset full_conservative --device 0,1 --baseline-fix --resume-missing
```

`yolo_dataset_grouped/val` 和 `test` 是 official 泛化评估集，不用于 early stopping。训练选择 checkpoint 时使用 `configs/data/bubble_train_dev.yaml`。

## 完成标准

完成训练相关任务前必须报告：

- 改动文件。
- 跑过的检查或训练命令。
- 生成的 summary/report 路径。
- 任何未验证项和风险。
