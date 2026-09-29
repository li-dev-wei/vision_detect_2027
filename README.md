# 基于改进 YOLO12 的锂电池 RGB 微小表面缺陷检测

本仓库用于复现和改进 YOLO12，研究普通 RGB 图像中的锂电池微小表面缺陷检测。项目优先解决微小目标在缩放和下采样过程中信息丢失的问题，并在准确率、召回率与推理速度之间取得平衡。

当前路线不是一次性堆叠模块，而是按照“数据审计 → 强基线 → P2 检测头 → 多尺度融合 → 定位损失”的顺序逐项验证。

## 当前范围

- 检测任务：有监督目标检测，标注格式为 YOLO 边界框。
- 基准模型：COCO 预训练的 `yolo12n.pt`。
- 重点指标：微小缺陷 Recall、mAP50-95、参数量、FLOPs 和端到端延迟。
- 首个结构改进：增加步长为 4 的 P2 检测分支。
- 后续实验：轻量多尺度融合、IoU 类损失与 NWD 组合损失。

YOLO12 默认使用 P3、P4、P5 三个检测尺度。本仓库提供的 P2 配置在保留原有三个尺度的同时，引入高分辨率浅层特征。

## 仓库结构

```text
vision_detect_2027/
├── configs/
│   ├── data/battery.yaml
│   └── models/yolo12n-p2.yaml
├── docs/
│   ├── ablation-plan.md
│   ├── algorithm-roadmap.md
│   └── annotation-guide.md
├── scripts/
│   ├── analyze_dataset.py
│   ├── evaluate.py
│   └── train_baseline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 环境安装

建议使用 Python 3.11 和带 CUDA 的 PyTorch 环境。先根据本机 CUDA 版本安装 PyTorch，再安装本项目依赖：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

首次形成稳定实验环境后，应执行 `pip freeze > environment-lock.txt`，并在实验记录中保存显卡型号、CUDA、PyTorch 和 Ultralytics 版本。

## 数据准备

推荐目录：

```text
datasets/battery/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每张图片对应一个同名 `.txt` 标签文件，每行格式为：

```text
class_id x_center y_center width height
```

坐标均归一化到 `[0, 1]`。无缺陷图片可以没有标签文件，也可以使用空标签文件。类别定义与划分要求见 [标注规范](docs/annotation-guide.md)。

以下命令均应在仓库根目录执行。正式训练前运行数据审计：

```bash
python scripts/analyze_dataset.py --data configs/data/battery.yaml --imgsz 1024 --check-duplicates
```

脚本会统计类别分布、缺失或错误标签、背景图比例、缩放后的目标尺寸，以及训练集、验证集和测试集之间的重复图片。

## 训练基线

先完成 3 个输入基准实验：640 整图、1024 整图和重叠切片。除输入方式外，其他训练参数保持一致。

```bash
python scripts/train_baseline.py \
  --data configs/data/battery.yaml \
  --model yolo12n.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch -1 \
  --device 0 \
  --name b1_yolo12n_1024
```

Windows PowerShell 可以将多行命令写成一行执行。

## 训练 P2 模型

P2 模型结构与基线不同，因此先创建模型，再加载能够匹配的 YOLO12n 预训练参数：

```bash
python scripts/train_baseline.py \
  --data configs/data/battery.yaml \
  --model configs/models/yolo12n-p2.yaml \
  --pretrained yolo12n.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch -1 \
  --device 0 \
  --name e1_yolo12n_p2_1024
```

新增 P2 分支会增加显存和计算量。如果显存不足，优先减小 `batch`，不要先降低输入尺寸。

## 评估

测试集只在模型和置信度策略确定后使用：

```bash
python scripts/evaluate.py \
  --model runs/battery/e1_yolo12n_p2_1024/weights/best.pt \
  --data configs/data/battery.yaml \
  --split test \
  --imgsz 1024 \
  --device 0
```

完整实验编号、变量控制和结果表见 [消融实验计划](docs/ablation-plan.md)。总体实现顺序见 [算法路线](docs/algorithm-roadmap.md)。

## 复现约定

- 不上传原始数据、训练权重、推理结果和 `runs/`。
- 数据集按电池编号、生产批次或采集序列分组划分，禁止相邻帧跨集合。
- 所有对比实验使用相同的数据版本、随机种子、输入尺寸和训练轮数。
- 筛选实验可运行一次；基线与最终模型至少运行 3 个随机种子并报告均值和标准差。
- 测试集不能用于选择网络结构、损失权重或置信度阈值。

## 参考实现

- [YOLOv12 作者仓库](https://github.com/sunsmarterjie/yolov12)
- [Ultralytics YOLO12 文档](https://docs.ultralytics.com/models/yolo12)
- [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)
- [A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/abs/2110.13389)
