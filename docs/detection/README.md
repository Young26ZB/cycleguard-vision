# NMS-Free 电动自行车检测指南

项目使用 Ultralytics YOLO11 实现端到端电动自行车检测。模型和损失面向拥挤、遮挡场景设计，实际性能仍待云端实验评估。

## 模型配置

| 配置 | 网络与用途 |
| --- | --- |
| [yolo11-ebike.yaml](../../ultralytics/cfg/models/11/yolo11-ebike.yaml) | 默认配置，使用标准 YOLO11 特征提取与融合结构和端到端检测头。做基础对照时显式传入 `--no-repulsion`。 |
| [yolo11-nms-free.yaml](../../ultralytics/cfg/models/11/yolo11-nms-free.yaml) | 增强配置，在骨干与检测特征中加入 SimAM，在 P5→P4 和 P4→P3 上采样中使用 DySample。 |

两种配置都支持 `--model-scale n/s/m/l/x`。BiFPN 模块已有实现，接入模型配置和消融实验仍待完成。

## 检测链路

1. 骨干和颈部网络提取并融合 P3、P4、P5 特征，由 [Detect](../../ultralytics/nn/modules/head.py) 输出分类和回归预测。
2. 训练时，[E2EDetectLoss](../../ultralytics/utils/loss.py) 分别对 `one2many` 和 `one2one` 分支计算损失并求和。两支使用 TaskAlignedAssigner，候选上限分别为 10 和 1。分配机制属于任务对齐分配，无法保证每个 GT 都获得匹配。
3. 启用 [Repulsion Loss](../../ultralytics/utils/repulsion_loss.py) 时，两支在同一张图的统一图像坐标中计算排斥项。RepGT 排除预测框匹配的 GT，RepBox 排除匹配同一个 GT 的预测框对。训练入口默认启用排斥损失，`--repulsion-weight` 默认值为 0.5。
4. 推理使用一对一分支，输出形状为 `(B, N, 6)`，每项包含 `[x1, y1, x2, y2, conf, cls]`。结果保留 TopK、置信度和最大检测数过滤，跳过基于 IoU 的 NMS。

[DySample](../../ultralytics/nn/modules/dysample.py) 在插值后学习内容相关的采样偏移。[SimAM](../../ultralytics/nn/modules/attention.py) 通过无可学习参数的注意力对特征重新加权。

## 环境与数据

在仓库根目录安装项目，确保 Python 导入的是本仓库中的 Ultralytics。

```bash
pip install -e .
python -c "import ultralytics; print(ultralytics.__file__)"
```

[数据配置](../../ultralytics/cfg/datasets/ebike.yaml) 预期使用 YOLO 检测标注。数据根目录下使用 `images/train`、`images/val` 及对应的 `labels/train`、`labels/val`，每行标注为归一化的 `class x_center y_center width height`。电动自行车类别 ID 为 0。数据位于其他位置时，通过 `--data-root` 指定实际路径。

## 推理与评估

以下命令从仓库根目录执行，将权重和输入路径替换成实际文件。

```bash
python inference_ebike.py --weights /path/to/best.pt --source image.jpg
python inference_ebike.py --weights /path/to/best.pt --source video.mp4
python inference_ebike.py --weights /path/to/best.pt --source 0 --show
python inference_ebike.py --weights /path/to/best.pt --source images/
```

推理结果默认写入 `runs/detect` 下的实验目录。可通过 `--save-txt` 保存标注，或通过 `--save-crop` 保存目标裁剪。

在阿里云使用真实数据评估时，可以运行以下命令。输出包括 `metrics.json` 和 `evaluation_report.txt`，安装 matplotlib 和 seaborn 后可使用 `--save-plots` 生成图表。

```bash
python evaluate_model.py --weights /path/to/best.pt --data-root /data/ebike --device 0 --save-plots
```

本地合成张量检查、阿里云训练命令和待验证项目见 [第一阶段实现与云端验证](STAGE1.md)。
