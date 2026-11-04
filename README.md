# CycleGuard Vision

面向非机动车的视觉感知、时空轨迹重构与可解释违章行为检测系统。当前主要实现基于 Ultralytics YOLO11 的 NMS-Free 电动自行车检测。非线性跟踪和神经符号违章推理仍处于设计阶段。

- [文档索引](docs/README.md)
- [检测指南](docs/detection/README.md)
- [第一阶段实现与云端验证](docs/detection/STAGE1.md)
- [OpenSpec 开发指南](docs/development/OPENSPEC_GUIDE.md)
- [仓库开发约定](AGENTS.md)

数据集、模型权重、训练结果和本地依赖需单独准备。真实数据训练与 CUDA 验证在阿里云执行。

本项目基于 Ultralytics，沿用 [AGPL-3.0 许可证](LICENSE)。上游框架的引用信息见 [CITATION.cff](CITATION.cff)，文档与示例见 [Ultralytics 文档](https://docs.ultralytics.com/) 和仓库内的 [examples](examples/)。
