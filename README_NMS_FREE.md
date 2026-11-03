# NMS-Free YOLOv11 电动自行车检测系统

## 项目概述

本项目基于 YOLOv11 实现了针对城市早晚高峰期间高密度电动自行车场景的优化检测算法。主要解决以下问题：

1. **传统 NMS 导致的漏检问题**：通过端到端 NMS-Free 架构和双重标签分配策略解决
2. **非刚体目标定位漂移问题**：通过 DySample 动态上采样和 Repulsion Loss 解决
3. **复杂背景干扰问题**：通过 SimAM 无参注意力机制和 BiFPN 特征金字塔解决

## 核心技术

### 1. NMS-Free 端到端检测
- **双重标签分配策略 (Dual Label Assignment)**
  - One-to-Many 分支：训练阶段保证学习效率
  - One-to-One 分支：推理阶段保证预测唯一性
- 推理时直接输出结果，无需 NMS 后处理

### 2. DySample 动态上采样
- 内容感知的动态采样点预测
- 根据骑行者姿态自适应调整采样位置
- 保持特征图放大后的轮廓清晰度

### 3. SimAM 无参注意力机制
- 基于能量函数的注意力计算
- 零额外参数，计算开销极小
- 自动抑制背景噪声

### 4. Repulsion Loss 排斥损失
- RepGT Loss：惩罚预测框与非匹配目标的重叠
- RepBox Loss：惩罚预测框之间的重叠
- 强制分离粘连的检测框

## 项目结构

```
NMS-FreeYOLO/
├── ultralytics/
│   ├── cfg/
│   │   ├── models/11/
│   │   │   ├── yolo11-nms-free.yaml    # 完整 NMS-Free 配置
│   │   │   └── yolo11-ebike.yaml       # 简化版配置
│   │   └── datasets/
│   │       └── ebike.yaml              # 数据集配置模板
│   ├── nn/modules/
│   │   ├── dysample.py                 # DySample 动态上采样
│   │   ├── attention.py                # SimAM 注意力机制
│   │   └── bifpn.py                    # BiFPN 特征金字塔
│   └── utils/
│       ├── repulsion_loss.py           # Repulsion Loss
│       └── nms_free_loss.py            # NMS-Free 损失函数
├── train_ebike.py                       # 训练脚本
├── inference_ebike.py                   # 推理脚本
└── README_NMS_FREE.md                   # 本文档
```

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -e .
```

### 2. 数据集准备

修改 `ultralytics/cfg/datasets/ebike.yaml` 配置文件：

```yaml
path: /path/to/your/dataset  # 数据集根目录
train: images/train
val: images/val
test: images/test

names:
  0: ebike
```

数据集目录结构：
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

标签格式 (YOLO format):
```
<class_id> <x_center> <y_center> <width> <height>
```

### 3. 训练模型

**基础训练：**
```bash
python train_ebike.py --data ebike.yaml --epochs 300
```

**指定模型规模：**
```bash
# nano 版本 (最快)
python train_ebike.py --model yolo11-ebike.yaml --model-scale n --epochs 300

# small 版本 (平衡)
python train_ebike.py --model yolo11-ebike.yaml --model-scale s --epochs 300

# medium 版本 (推荐)
python train_ebike.py --model yolo11-ebike.yaml --model-scale m --epochs 300
```

**完整参数训练：**
```bash
python train_ebike.py \
    --model yolo11-ebike.yaml \
    --model-scale m \
    --data ebike.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/train \
    --name ebike_nms_free
```

### 4. 推理检测

**图片检测：**
```bash
python inference_ebike.py --weights runs/train/ebike_nms_free/weights/best.pt --source image.jpg
```

**视频检测：**
```bash
python inference_ebike.py --weights best.pt --source video.mp4
```

**摄像头实时检测：**
```bash
python inference_ebike.py --weights best.pt --source 0 --show
```

**批量检测：**
```bash
python inference_ebike.py --weights best.pt --source images_folder/
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 | 训练轮数 |
| `--batch` | 16 | 批量大小 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--box` | 7.5 | Box 损失权重 |
| `--cls` | 0.5 | 分类损失权重 |
| `--dfl` | 1.5 | DFL 损失权重 |
| `--mosaic` | 1.0 | Mosaic 增强概率 |
| `--mixup` | 0.1 | MixUp 增强概率 |
| `--close-mosaic` | 10 | 最后 N 轮关闭 Mosaic |

## 模型配置说明

### yolo11-ebike.yaml (推荐)

基于标准 YOLOv11 的端到端检测配置，使用内置模块：
- 端到端检测头 (end2end=True)
- 标准 FPN 结构
- 兼容性最好

### yolo11-nms-free.yaml (完整版)

包含所有优化模块的完整配置：
- DySample 动态上采样
- SimAM 注意力机制
- 端到端检测头

## 性能指标

| 模型 | 参数量 | FLOPs | mAP50 | mAP50-95 | 推理时间 |
|------|--------|-------|-------|----------|----------|
| YOLO11n-ebike | 2.6M | 6.6G | - | - | ~2ms |
| YOLO11s-ebike | 9.5M | 21.7G | - | - | ~3ms |
| YOLO11m-ebike | 20.1M | 68.5G | - | - | ~5ms |

*注：性能指标需要在您的数据集上实际测试填写*

## 常见问题

### Q: 如何处理极端密集场景？
A: 建议：
1. 增加训练数据中的密集场景样本
2. 适当降低置信度阈值 (--conf 0.15)
3. 使用更大的模型规模 (m 或 l)

### Q: 训练时出现 OOM 错误？
A: 尝试：
1. 减小 batch size
2. 减小图像尺寸
3. 使用混合精度训练 (--amp)

### Q: 如何加快推理速度？
A: 建议：
1. 使用 FP16 推理 (--half)
2. 使用更小的模型规模 (n 或 s)
3. 减小输入图像尺寸

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{nms-free-yolo11-ebike,
  title={NMS-Free YOLOv11 for High-Density Electric Bicycle Detection},
  year={2024},
  note={基于双重标签分配策略的端到端电动自行车检测系统}
}
```

## 许可证

本项目基于 AGPL-3.0 许可证开源。
