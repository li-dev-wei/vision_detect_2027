# 模型配置状态

申报书明确以 YOLO26n 为基线。本目录的配置均基于 Ultralytics 官方 YOLO26 网络结构：

- `yolo26n-p2.yaml`：官方 P2/4～P5/32 四尺度结构，类别数改为 18；
- `yolo26n-dmare.yaml`：标准 P3/8～P5/32 结构，在 P3 融合节点后加入 DMARE；
- `yolo26n-p2-dmare.yaml`：官方 P2 结构，在 P2 融合节点后加入 DMARE。

三份 YAML 都保留 YOLO26 的 `end2end: true`、`reg_max: 1`、新版 SPPF 参数及检测头结构。结构实验使用 `yolo26n.pt` 加载能够匹配的预训练权重，并记录实际匹配率。
