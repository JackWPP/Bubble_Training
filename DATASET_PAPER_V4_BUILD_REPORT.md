# Bubble YOLO Paper V4 Dataset Build Report

## Benchmark Definition

Paper V4 is a source-key stratified random split / in-distribution benchmark.
It is designed for the main paper training curves and baseline selection. It is
not a strict cross-domain or grouped OOD benchmark.

- output directory: `G:/Bubble_Train/yolo_dataset_paper_v4`
- data config: `G:/Bubble_Train/yolo_dataset_paper_v4/bubble.yaml`
- split mode: `paper-v4`
- seed: `42`
- target split: train/val/test = 0.7/0.15/0.15
- train augmentation: base samples + horizontal flip only
- validation/test augmentation: none
- leakage rule: `source_key` cannot cross split
- intentional relaxation: `source` and `group_key` may cross split
- external stress test: report `yolo_dataset_grouped` separately as grouped OOD

## Raw Source Summary

| Source | Valid images | Valid boxes | Missing images | Invalid boxes | Common dimensions |
| --- | ---: | ---: | ---: | ---: | --- |
| 20+40 | 66 | 2233 | 0 | 0 | 502x404(34), 502x400(32) |
| 60+80 | 47 | 442 | 0 | 0 | 508x426(24), 504x424(23) |
| big_fengchao | 6 | 1460 | 0 | 0 | 1920x1080(6) |
| bubble_1 | 3 | 590 | 0 | 0 | 1920x1080(3) |
| bubble_fc | 12 | 602 | 0 | 0 | 480x480(11), 1080x1080(1) |
| bubble_pad | 3 | 1484 | 0 | 0 | 1920x1080(3) |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0 | 33 | 2713 | 0 | 0 | 1920x1080(33) |

Raw categories: `{'bubble': 9524}`

## Processing Strategy

- Large images are tiled with 640x640 sliding windows, stride 480.
- Small images are letterboxed into 640x640 while preserving aspect ratio.
- A retained tile box must keep its center in the tile, have width/height at
  least 4 px after clipping, and retain at least 40% of the original area.
- Offline augmentation is train-only and limited to horizontal flip.
- `manifest.json` records both `source_key` and `group_key` for leakage review.

## Output Summary

| Split | Images | Label files | Boxes | Base samples | Augmented samples | Boxes/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 644 | 644 | 17002 | 322 | 322 | 26.40 |
| val | 83 | 83 | 2731 | 83 | 0 | 32.90 |
| test | 83 | 83 | 2724 | 83 | 0 | 32.82 |

Box width statistics: `{'count': 22457, 'min': 6.31, 'p50': 45.16, 'p90': 76.362, 'max': 312.4, 'mean': 47.164}`

Box height statistics: `{'count': 22457, 'min': 5.87, 'p50': 43.55, 'p90': 74.48, 'max': 305.81, 'mean': 46.165}`

## Quality Checks

- source_key split leakage count: `0`
- group_key split overlap count: `11` (intentional)
- augmented val/test samples: `0`
- validation result: `PASS` with `0` errors

## Source Coverage

| Source | Train images | Val images | Test images | Total images | Total boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20+40 | 92 | 10 | 10 | 112 | 3877 |
| 60+80 | 66 | 7 | 7 | 80 | 730 |
| big_fengchao | 64 | 8 | 8 | 80 | 4243 |
| bubble_1 | 16 | 8 | 8 | 32 | 1472 |
| bubble_fc | 22 | 2 | 2 | 26 | 1068 |
| bubble_pad | 16 | 8 | 8 | 32 | 3545 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0 | 368 | 40 | 40 | 448 | 7522 |

## Group Coverage

| group_key | Train images | Val images | Test images | Total images | Total boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20+40|capture_series | 92 | 10 | 10 | 112 | 3877 |
| 60+80|capture_series | 66 | 7 | 7 | 80 | 730 |
| big_fengchao|video_or_scene | 64 | 8 | 8 | 80 | 4243 |
| bubble_1|video_or_scene | 16 | 8 | 8 | 32 | 1472 |
| bubble_fc|-40-0.16A-1(480P).mp4 | 6 | 0 | 0 | 6 | 292 |
| bubble_fc|-40-0.2A-1(480p).mp4 | 4 | 1 | 1 | 6 | 308 |
| bubble_fc|-40-0.2A-1.mp4 | 8 | 0 | 0 | 8 | 170 |
| bubble_fc|-40-0.4A-1(480P).mp4 | 4 | 1 | 1 | 6 | 298 |
| bubble_pad|video_or_scene | 16 | 8 | 8 | 32 | 3545 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.04-1 | 112 | 16 | 8 | 136 | 2877 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.08-2 | 144 | 8 | 0 | 152 | 1419 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.12-1 | 48 | 8 | 16 | 72 | 1730 |
| job_13_dataset_2026_04_30_19_34_23_coco 1.0|0.16-2 | 64 | 8 | 16 | 88 | 1496 |

## Paper Wording

Use Paper V4 as the source-key stratified in-distribution benchmark for the
main baseline curves. Do not describe it as a strict OOD split. Use
`yolo_dataset_grouped` as the external grouped OOD stress test and report it in
a separate generalization section.
