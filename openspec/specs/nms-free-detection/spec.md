# NMS-Free Detection

## Purpose

记录第一阶段已实现的检测能力基线。此规范描述代码行为，不承诺拥堵场景性能收益；现有验证范围见 STAGE1.md。

## Requirements

### Requirement: Explicit model selection
训练入口 SHALL 使用指定的 YOLO11 架构和模型规模，找不到配置时 MUST 报错而非回退到另一架构。

#### Scenario: Choose a model scale
- **WHEN** 使用 yolo11-ebike.yaml 并指定 model-scale=s
- **THEN** 实际构建并在模型配置中保留 s 规模

#### Scenario: Reject unknown configuration
- **WHEN** 提供不存在的模型配置名称
- **THEN** 抛出明确错误，不改用标准 YOLO11 继续运行

### Requirement: End-to-end detection branches
端到端检测器 SHALL 使用一对多与一对一分支训练，在推理中使用一对一预测，保留置信度和 TopK 过滤而跳过 IoU NMS。

#### Scenario: Infer with an end-to-end model
- **WHEN** 端到端模型执行图像前向推理
- **THEN** 输出批次检测结果，每项含 xyxy、置信度和类别，后处理不执行基于 IoU 的抑制

### Requirement: Matching-aware repulsion
启用排斥损失时，检测损失 MUST 在每张图的统一图像坐标中计算 RepGT 与 RepBox，排除本目标 GT 及同 GT 预测对。

#### Scenario: Multiple positives for one target
- **WHEN** 多个前景预测匹配同一 GT 且图中没有其他 GT
- **THEN** 不因这些预测之间的重叠产生排斥损失

#### Scenario: Predictions at different strides
- **WHEN** 来自不同特征层的预测参与排斥计算
- **THEN** 先转换到统一图像坐标，再计算框间关系

#### Scenario: Disable repulsion for a baseline
- **WHEN** 训练入口指定 no-repulsion
- **THEN** 排斥损失项为零，保留双分支分类与回归监督

### Requirement: Consistent dynamic sampling grid
动态上采样 SHALL 使用与 align_corners=False 一致的像素中心采样坐标，并支持有限数值的合成数据反向传播。

#### Scenario: Zero learned offsets
- **WHEN** DySample 偏移为零
- **THEN** 结果在数值误差范围内等于该模块的初始双线性上采样结果

### Requirement: Portable dataset configuration
训练及评估入口 SHALL 支持显式指定数据根目录，并保存解析后的配置副本。

#### Scenario: Move to the cloud
- **WHEN** 用户通过 data-root 指定阿里云上的数据目录
- **THEN** 使用该目录解析训练与验证输入，不改写源数据配置，不依赖开发机路径
