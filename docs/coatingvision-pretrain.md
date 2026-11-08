# CoatingVision 极片涂布预训练

## 数据选择

CoatingVision 当前包含 2,227 张 RGB 分类图像和 581 个目标检测标签文件。分类表有 Surface_Crack、Delamination、Pinhole、unclassified 四个图像级标签；检测框只出现类别 0 和 1，且未找到类别定义文档。

为避免把不确定的检测类别用于训练，预训练子集只保留同时满足以下条件的样本：

- 图像级标签只有 Pinhole；
- 目标检测标签只含类别 0；
- 分类表、检测标签和 RGB 图像均存在。

从源文件名中提取采集组，按组留出验证数据，避免同一采集序列的图像同时进入训练集和验证集。类别 0 映射为 `electrode_pinhole` 是根据标签共现关系作出的保守推断，不代表已获得原数据集官方类别说明。

## 准备数据

```bash
python scripts/prepare_coatingvision_pretrain.py
```

脚本将生成 `dataset/CoatingVision/pretrain_pinhole/`，不修改原数据。输出目录包含单类 YOLO 数据配置、分组清单和来源 manifest。

## 预训练

环境可导入 Ultralytics 后，先用较短训练作为迁移初始化：

```bash
python scripts/train_baseline.py \
  --data dataset/CoatingVision/pretrain_pinhole/data.yaml \
  --model yolo26n.pt \
  --imgsz 640 \
  --epochs 100 \
  --patience 20 \
  --batch 8 \
  --workers 2 \
  --device 0 \
  --project runs/coatingvision-pretrain \
  --name yolo26n_pinhole_rgb
```

这是迁移学习候选初始化，不是最终指标实验。CoatingVision 子集较小，应把 COCO 初始化和该预训练初始化作为平行方案，在同一 18 类目标数据集上微调比较；最终只根据冻结测试集的结果判断是否保留。

## 迁移到 18 类模型

对基线可用 `--model yolo26n.pt --pretrained runs/coatingvision-pretrain/yolo26n_pinhole_rgb/weights/best.pt`。对 P2/DMARE 配置，将 `--model` 指向对应 YAML，`--pretrained` 仍指向该预训练权重。不同类别数或新增模块导致部分参数无法匹配，这是预期行为；骨干和形状匹配的层会用于初始化。
