# Server Upload Package Manifest

## 需要上传到服务器的文件

### 1. SAM3 推理脚本（segmentation/ 目录）
```
segmentation/
├── generate_masks.py           # ★ 核心脚本
├── HANDOFF_SAM3_SERVER.md      # 详细操作指南
├── test_sam3_wsl.py            # 环境验证脚本
├── debug_single_image.py       # 单图耗时分析
├── test_sam3_concept.py        # 概念匹配测试
└── test_sam3_resize.py         # 大图缩放测试
```

### 2. 数据集（Dataset/ 目录）
```
Dataset/
├── 20+40/                       # 已完成 SAM3 推理（可跳过）
│   └── annotations/
│       ├── instances_default.json
│       └── instances_default_segmented.json  # SAM3 已生成
├── 60+80/                       # 已完成 SAM3 推理（可跳过）
│   └── annotations/
│       ├── instances_default.json
│       └── instances_default_segmented.json  # SAM3 已生成
├── big_fengchao/                # ⏳ 待处理
│   └── annotations/
│       └── instances_default.json
├── bubble_1/                    # ⏳ 待处理
├── bubble_fc/                   # ⏳ 待处理
├── bubble_pad/                  # ⏳ 待处理
└── job_13_dataset_2026_04_30_19_34_23_coco 1.0/  # ⏳ 待处理
```

每个数据源必须保留完整目录结构：
```
Dataset/<source_name>/
├── annotations/
│   └── instances_default.json
└── images/
    └── default/
        └── *.jpg
```

### 3. SAM3 模型权重

下载地址: https://huggingface.co/facebook/sam3

需要的文件（HuggingFace 格式）：
```
sam3/
├── model.safetensors       # 3.4GB ★
├── config.json
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── special_tokens_map.json
```

## 上传后在服务器上的目录结构

```
/path/to/server/Bubble_Train/
├── segmentation/              # 本目录的脚本文件
│   └── ... (上面列出的 .py 文件)
├── Dataset/                   # 数据（全部 7 个源）
│   └── ... (上面列出的目录结构)
└── sam3/                      # SAM3 权重
    └── model.safetensors

# 修改 generate_masks.py 中的路径：
# MODEL_PATH = "/path/to/server/sam3"
# DATASET_DIR = Path("/path/to/server/Bubble_Train/Dataset")
```

## 服务器的第一步

```bash
cd /path/to/server/Bubble_Train/segmentation
# 1. 安装依赖
pip install torch transformers accelerate opencv-python pillow shapely tqdm

# 2. 修改 generate_masks.py 中的路径
vim generate_masks.py  # MODEL_PATH = "...", DATASET_DIR = Path("...")

# 3. 验证
python test_sam3_wsl.py  # 注意：路径也需要改成服务器路径

# 4. 运行
python generate_masks.py
```
