# 基于改进 YOLO26n 的新能源表面微小缺陷检测

本仓库围绕申报书明确指定的 YOLO26n 基线，研究新能源材料 RGB 图像中的微小表面缺陷检测。模型改进聚焦于 P2 高分辨率检测分支和缺陷形态感知特征增强模块（DMARE）。

研究顺序为“确认 18 类 RGB 映射与数据 → 建立 YOLO26n 基线 → P2 和 DMARE 消融 → 误差分析 → 边缘部署评估”。

## 当前范围

- 检测任务：有监督目标检测，标注格式为 YOLO 边界框。
- 基准模型：COCO 预训练的 `yolo26n.pt`。
- 重点指标：微小缺陷 Recall、mAP50-95、参数量、FLOPs 和端到端延迟。
- 首个结构改进：增加步长为 4 的 P2 检测分支。
- 形态增强：DMARE 使用 7/11 两种尺度的水平、垂直、45°、135°方向分支、局部分支和位置自适应门控融合。

YOLO26 标准模型使用 P3/8、P4/16、P5/32 三个检测尺度；官方 P2 变体增加 P2/4 输出。本仓库直接以官方 YOLO26 P2 配置为结构基准，再分别接入 DMARE，保留 YOLO26 的端到端双检测头与 `reg_max: 1` 设置。

## 仓库结构

```text
vision_detect_2027/
├── configs/
│   ├── data/battery.yaml
│   ├── data/class-mapping-template.csv
│   └── models/
│       ├── README.md
│       ├── yolo26n-p2.yaml
│       ├── yolo26n-dmare.yaml
│       └── yolo26n-p2-dmare.yaml
├── docs/
│   ├── ablation-plan.md
│   ├── algorithm-roadmap.md
│   └── annotation-guide.md
│   └── coatingvision-pretrain.md
├── scripts/
│   ├── analyze_dataset.py
│   ├── check_environment.py
│   ├── prepare_coatingvision_pretrain.py
│   ├── evaluate.py
│   └── train_baseline.py
├── .gitignore
├── README.md
├── requirements.txt
└── models/
    ├── dmare.py
    └── ultralytics_plugins.py
```

## 环境安装

训练需要与设备匹配的 PyTorch/CUDA 环境。Jetson 上优先保留 JetPack 提供的 PyTorch，不要直接用 pip 升级 torch 或 NumPy。当前 `requirements.txt` 尚未在本机 ARM64 环境验证；先运行环境检查并解决不兼容项，再安装项目依赖：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python scripts/check_environment.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/check_environment.py
```

环境检查除验证依赖导入外，还会检查已安装的 Ultralytics 包是否包含 YOLO26 配置。首次形成稳定环境后，执行 `pip freeze > environment-lock.txt`，并记录 JetPack、CUDA、PyTorch 和 Ultralytics 版本。

## 数据准备

申报书将研究范围固定为 18 类 RGB 缺陷。当前映射从资料末尾汇总表中排除“光伏电池组件封装异常”和“泡沫镍局部压塌”，其余光伏电池片与泡沫镍缺陷保留。映射见 [类别表](configs/data/class-mapping-template.csv)。

推荐目录：

```text
dataset/new_energy_rgb/
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

脚本会统计类别分布、缺失或错误标签、背景图比例、缩放后的短边尺寸、申报书规定的 COCO 面积分级，以及训练集、验证集和测试集之间的重复图片。

## 训练基线

先完成 3 个输入基准实验：640 整图、1024 整图和重叠切片。除输入方式外，其他训练参数保持一致。

```bash
python scripts/train_baseline.py \
  --data configs/data/battery.yaml \
  --model yolo26n.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch -1 \
  --device 0 \
  --name b1_yolo26n_1024
```

Windows PowerShell 可以将多行命令写成一行执行。

## 训练 P2 与 DMARE 模型

结构模型先由 YAML 创建，再加载形状匹配的 YOLO26n 预训练参数：

```bash
python scripts/train_baseline.py \
  --data configs/data/battery.yaml \
  --model configs/models/yolo26n-p2.yaml \
  --pretrained yolo26n.pt \
  --imgsz 1024 \
  --epochs 200 \
  --batch -1 \
  --device 0 \
  --name e1_yolo26n_p2_1024
```

新增 P2 分支会增加显存和计算量；显存不足时优先减小 `batch`，不要先降低输入尺寸。

## 评估

测试集只在模型和置信度策略确定后使用：

```bash
python scripts/evaluate.py \
  --model runs/battery/e1_yolo26n_p2_1024/weights/best.pt \
  --data configs/data/battery.yaml \
  --split test \
  --imgsz 1024 \
  --device 0
```

完整实验编号、变量控制和结果表见 [消融实验计划](docs/ablation-plan.md)。总体实现顺序见 [算法路线](docs/algorithm-roadmap.md)。

四组核心实验为 `yolo26n.pt`、`yolo26n-p2.yaml`、`yolo26n-dmare.yaml` 和 `yolo26n-p2-dmare.yaml`。后三组使用 `--pretrained yolo26n.pt` 加载形状匹配的预训练参数；DMARE 配置会由训练脚本自动注册自定义模块。

```bash
python scripts/train_baseline.py --data configs/data/battery.yaml --model configs/models/yolo26n-dmare.yaml --pretrained yolo26n.pt --imgsz 1024 --epochs 200 --batch -1 --device 0 --name e2_yolo26n_dmare
python scripts/train_baseline.py --data configs/data/battery.yaml --model configs/models/yolo26n-p2-dmare.yaml --pretrained yolo26n.pt --imgsz 1024 --epochs 200 --batch -1 --device 0 --name e3_yolo26n_p2_dmare
```

## 复现约定

- 不上传原始数据、训练权重、推理结果和 `runs/`。
- 数据集按电池编号、生产批次或采集序列分组划分，禁止相邻帧跨集合。
- 所有对比实验使用相同的数据版本、随机种子、输入尺寸和训练轮数。
- 筛选实验可运行一次；基线与最终模型至少运行 3 个随机种子并报告均值和标准差。
- 测试集不能用于选择网络结构、损失权重或置信度阈值。

## 参考实现

- [Ultralytics 官方仓库](https://github.com/ultralytics/ultralytics)
- [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26)
- [A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/abs/2110.13389)
