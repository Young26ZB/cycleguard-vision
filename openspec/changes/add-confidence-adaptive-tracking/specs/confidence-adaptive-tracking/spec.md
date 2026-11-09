# Spec Delta

## Purpose

为真实检测引入置信度自适应的观测更新，并用同一检测输入、质量门槛与时延预算验收第二部分的组合配置，使候选部署方案可比较、可复现。

## ADDED Requirements

### Requirement: Confidence-dependent real observation updates

启用 NSA 时，跟踪器 SHALL 对已匹配的真实检测按置信度非增地调整观测噪声，并保持正的噪声下限。该开关 SHALL 可单独控制，观测框仍用原图像素表达。ORU 虚拟观测沿用其独立噪声规则。

#### Scenario: Compare two confidence scores for the same detection box

- **WHEN** 同一个预测状态和观测框分别使用有效低分与高分执行更新，且两者未同时落入同一噪声限幅区间
- **THEN** 低分更新采用更大的观测噪声。对单个独立观测维度施加相同非零残差时，低分引起的校正更小

#### Scenario: Disable NSA or process a virtual observation

- **WHEN** 关闭 NSA，或者 ORU 正在更新一条虚拟观测
- **THEN** 该次观测沿用对应的原噪声规则，不取虚拟框的检测分数，也不添加第二次真实更新

### Requirement: Stable bounded update cost

NSA SHALL 在既有真实更新中调整噪声，保持固定维度和每个匹配一次更新。更新 MUST 检查数值有效性，对越界或非有限检测分数沿用主链路输入校验。合法输入触发自适应数值失败时，跟踪器 SHALL 使用普通更新尝试恢复并记录回退。

#### Scenario: Confidence approaches one

- **WHEN** 合法检测分数接近或等于 1
- **THEN** 观测噪声保留正下限，输出状态和协方差满足有限性与数值稳定性检查

#### Scenario: Benchmark real measurement updates

- **WHEN** 在固定的高密度检测缓存上仅切换 NSA
- **THEN** 报告真实更新计数、NSA 回退次数以及 P95/P99，且开启组不为每个候选重复执行滤波。超过跟踪预算的配置不得标为实时配置

### Requirement: Controlled accuracy and latency comparison

组合评估 SHALL 固定权重、检测缓存、视频划分和测量环境，至少比较主链路、单独关闭 OCM、OCR、ORU 或 NSA 的组合以及全开配置。独立标注视频 SHALL 按序列划分验证集和测试集，并记录 IDF1、AssA、ID 切换、分遮挡时长的同 ID 恢复率及错误重连率。

#### Scenario: A target reappears without an eligible detection

- **WHEN** 评估带跨帧 ID 的遮挡重现事件
- **THEN** 分别报告全部可判定重现事件与其中有可匹配检测的事件结果，并记录分母，区分检测漏失与跟踪关联错误

#### Scenario: A quality threshold is used for selection

- **WHEN** 比较候选配置的准确率与时延
- **THEN** 先在验证阶段确定质量门槛、时延预算和候选配置，再在独立测试集按同一协议报告，避免根据测试结果修改阈值

### Requirement: Deploy only a qualified real-time profile

实时性验收 SHALL 沿用主链路的跟踪及端到端计时边界，报告 P50、P95、P99、吞吐、积压、掉帧和容量回退。在满足预定质量门槛、跟踪 P95 预算和目标帧周期约束的候选中，SHALL 优先选择端到端 P95 更低的配置。

#### Scenario: An enhancement misses its timing budget

- **WHEN** 某项增强在目标环境中超预算、造成持续积压、未处理完整帧序列或未达到质量门槛
- **THEN** 该配置不能作为已验收的实时配置，报告标明失败原因并比较关闭该功能或降低其开销的候选

#### Scenario: Real video or target hardware is unavailable

- **WHEN** 仅有本地合成回放、缺少独立标注视频或缺少目标环境端到端测量
- **THEN** 报告区分已完成的逻辑检查和待验证的质量或实时性结论，不将缓存回放吞吐计为端到端视频 FPS
