# HANDOFF: Paper V4 Baseline, Structure Ablation, and SSB Rescue

## Current State

This session moved the project from Paper V4 baseline selection into structure ablation and then into a targeted rescue plan for SSB-P3.

Current best single-run baseline remains:

- Run: `PV4S_768_LR0010_yolo11s_paper_v4`
- Path in downloaded results: `temp/v4/PV4S_768_LR0010_yolo11s_paper_v4/`
- Main test: `P=0.7841`, `R=0.7423`, `mAP50=0.7980`, `mAP50-95=0.4170`
- OOD grouped test: `P=0.8316`, `R=0.8107`, `mAP50=0.8740`, `mAP50-95=0.5528`
- Conf sweep best: `conf=0.35`, `F1=0.7633`
- Curve: best epoch `19`, best-to-last mAP50 drop `0.0263`, no NaN/Inf

The 832/896 resolution attempts did not replace this baseline:

- `PV4S_832_LR0010_E35_seed42`: smoother curve but lower main mAP50 `0.7875`
- `PV4S_896_LR0008_E35_seed42`: lower main mAP50 `0.7817`, OOD mAP50 `0.8481`, below the `0.85` threshold

## Completed Code Changes

The following changes have been implemented and verified locally.

### Paper V4 structure ablation

- Added experiment IDs in `configs/train/experiments.yaml`:
  - `PV4E1_SSB_P3`
  - `PV4E2_GLRB_P3`
  - `PV4E3_SSB_GLRB_P3`
- Added `scripts/run_nightly.py --paper-v4-ablation`.
- Fixed `tools/collect_results.py` so it collects all run directories containing `summary.json` or `results.csv`, including `PV4*`.

### DDP custom module fix

The first formal structure ablation failed under DDP with:

```text
KeyError: 'SSBRefine'
```

Cause: Ultralytics DDP temp worker scripts did not run `register_bubble_modules()`, so `parse_model()` in child processes could not resolve `SSBRefine`.

Fix:

- Added `ultralytics_custom/trainer.py`
  - `BubbleDetectionTrainer`
  - `BubbleNWDDetectionTrainer`
- Updated `scripts/train_experiment.py` to:
  - inject repo root into `PYTHONPATH` for DDP temp scripts
  - use `BubbleDetectionTrainer` for training
  - preserve NWD behavior through `BubbleNWDDetectionTrainer`

### Structure rescue plan implementation

Added identity-init SSB rescue artifacts:

- `configs/models/bubble_yolo11s_ssb_p3_identity.yaml`
  - Based on `bubble_yolo11s_ssb_p3.yaml`
  - SSB line changed to:
    ```yaml
    - [-1, 1, SSBRefine, [128, 1.0, True, 0.0]]
    ```
  - `gamma_init=0.0`, so initial forward is identity-like: `x + 0 * refine(x)`

Added training configs:

- `configs/train/paper_v4_768_lr0005_warm5.yaml`
  - `lr0=0.0005`
  - `warmup_epochs=5.0`
  - otherwise aligned to Paper V4 768 baseline
- `configs/train/paper_v4_768_lr0010_freeze17.yaml`
  - `lr0=0.0010`
  - `freeze=17`
  - otherwise aligned to Paper V4 768 baseline

Added rescue experiment IDs:

- `PV4E1Z_SSB_P3_ID`
- `PV4E1Z_SSB_P3_LOWLR`
- `PV4E1Z_SSB_P3_FREEZE17`

Added batch entry:

- `scripts/run_nightly.py --paper-v4-rescue`
  - Runs the three `PV4E1Z_*` experiments
  - Requires `--project runs/bubble_paper_v4`

Added curve diagnostics:

- `scripts/train_experiment.py` now records:
  - `first_map50`
  - `map50_gain_first_to_best`
  - first epoch P/R/mAP50-95
  - first epoch train/val DFL loss
- `tools/collect_results.py` and `tools/export_report.py` now include first mAP50 and gain columns.

## Current Experimental Evidence

Downloaded formal structure ablation results are in `temp/`:

- `temp/PV4E1_SSB_P3_yolo11s_paper_v4/`
- `temp/PV4E2_GLRB_P3_yolo11s_paper_v4/`
- `temp/PV4E3_SSB_GLRB_P3_yolo11s_paper_v4/`
- Summary table: `temp/experiment_summary.csv`
- Report: `temp/TRAINING_REPORT.md`

Results:

| Run | Main mAP50 | P | R | Conf best F1 | OOD mAP50 | Curve drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PV4S_768_LR0010` seed42 | `0.7980` | `0.7841` | `0.7423` | `0.7633` | `0.8740` | `0.0263` |
| `PV4E1_SSB_P3` | `0.7530` | `0.7700` | `0.7116` | `0.7403` | `0.8753` | `0.0022` |
| `PV4E2_GLRB_P3` | `0.7284` | `0.7586` | `0.6853` | `0.7208` | `0.8600` | `0.0041` |
| `PV4E3_SSB_GLRB_P3` | `0.7376` | `0.7929` | `0.6876` | `0.7365` | `0.8675` | `0.0057` |

Interpretation:

- Do not call this a hard model failure.
- E1/E2/E3 completed 50 epochs without NaN/Inf and have stable curves.
- But they start from a much worse transfer point:
  - baseline first epoch mAP50: `0.2573`
  - E1/E2/E3 first epoch mAP50: about `0.026`
- Most likely cause: adding P3 modules disrupted pretrained feature transfer; baseline training strategy is not appropriate for modified structures.
- Main test recall suffered most, which is bad for the target of reducing missed detections.
- E1 OOD mAP50 stayed strong, so the structure is not completely useless.

## Verified Commands Already Run Locally

Local verification passed:

```bash
python -m py_compile scripts/train_experiment.py scripts/run_nightly.py tools/collect_results.py tools/export_report.py
python -m compileall scripts tools ultralytics_custom
python -m pytest tests/test_bubble_modules.py
python tools/check_model_forward.py --model configs/models/bubble_yolo11s_ssb_p3_identity.yaml --imgsz 768 --device cpu
```

Also verified:

```bash
python tools/collect_results.py --project temp --out-csv temp\_rescue_collect_check.csv --out-json temp\_rescue_collect_check.json
python tools/export_report.py --project temp --summary-json temp\_rescue_collect_check.json --out temp\_rescue_report_check.md
```

The temporary `_rescue_*` files were removed.

## Next Agent: Recommended Immediate Steps

Do not rerun the old `PV4E1/PV4E2/PV4E3` ablation first. The next step is to run the SSB rescue experiments.

On the server:

```bash
cd /home/xgx/Bubble_Training
```

After syncing this repo state to the server, run:

```bash
python -m py_compile scripts/train_experiment.py scripts/run_nightly.py tools/collect_results.py tools/export_report.py
python -m compileall scripts tools ultralytics_custom
python -m pytest tests/test_bubble_modules.py
python tools/check_model_forward.py --model configs/models/bubble_yolo11s_ssb_p3_identity.yaml --imgsz 768 --device cuda:0
```

Smoke:

```bash
python scripts/train_experiment.py --exp PV4E1Z_SSB_P3_ID --device 0 --epochs 2 --batch 2 --workers 2 --project runs/bubble_paper_v4_rescue_smoke --name smoke_PV4E1Z_SSB_P3_ID --exist-ok --skip-val --skip-predict
```

Formal rescue:

```bash
python scripts/run_nightly.py --paper-v4-rescue --project runs/bubble_paper_v4 --device 0,1 --exist-ok --resume-missing --keep-going
```

Collect/report:

```bash
python tools/collect_results.py --project runs/bubble_paper_v4
python tools/export_report.py --project runs/bubble_paper_v4
```

If a run completes training but postprocess fails, rerun postprocess for that run:

```bash
python scripts/train_experiment.py --exp PV4E1Z_SSB_P3_ID --device 0,1 --exist-ok --postprocess-only
python scripts/train_experiment.py --exp PV4E1Z_SSB_P3_LOWLR --device 0,1 --exist-ok --postprocess-only
python scripts/train_experiment.py --exp PV4E1Z_SSB_P3_FREEZE17 --device 0,1 --exist-ok --postprocess-only
```

## Decision Criteria for Rescue Results

Primary question:

- Did identity init / lower LR / freeze fix the first-epoch transfer collapse?

Look first at:

- `curve_first_map50`
- `curve_map50_gain_first_to_best`
- main test mAP50
- conf best F1
- OOD test mAP50

Old bad reference:

- Old E1 first epoch mAP50: about `0.026`
- Old E1 main test mAP50: `0.7530`

Success criteria:

- Strong success:
  - any E1Z main test mAP50 `>= 0.790`
  - conf best F1 `>= 0.758`
  - OOD test mAP50 `>= 0.85`
- Weak success:
  - main test mAP50 improves at least `+0.025` over old E1, i.e. `>= 0.778`
  - first epoch mAP50 is clearly above old E1 `~0.026`

Decision rules:

- If `PV4E1Z_SSB_P3_ID` improves clearly, the issue was non-identity initialization. Next step: create and run E3 identity.
- If `PV4E1Z_SSB_P3_LOWLR` wins, the issue was too aggressive full fine-tuning. Next step: E3 with `lr0=0.0005`, `warmup=5`.
- If `PV4E1Z_SSB_P3_FREEZE17` wins, the issue was insufficient protection of pretrained feature path. Next step: E3 with freeze.
- If all three stay below `0.78`, stop the structure route and keep `PV4S_768_LR0010_yolo11s_paper_v4` as the paper baseline.

## Important Context and Risks

- Paper V4 is the thesis-friendly in-distribution benchmark.
- `yolo_dataset_grouped` is only OOD stress test, not checkpoint selection.
- Primary metric is main test mAP50, with Precision/Recall balance and conf best F1 as tie-breaks.
- mAP50-95 is diagnostic only; dense small/fuzzy bubbles make strict localization hard.
- Current goal is still a YOLO bubble detector baseline, not SAM/SAM3 integration.
- Do not change dataset split or labels in the next step.
- Do not start E4/E5/NWD until rescue results are understood.

